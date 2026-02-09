#!/usr/bin/env python3
"""
wrfout_to_wpsinit.py

Convert WRF wrfout NetCDF history files to WPS "intermediate" files readable by metgrid.exe.

Fixes:
- Ensure ALL 2D fields use level="200100" (string). Your pywinter expects len(level) to work.
- Generate soil inputs REQUIRED by your METGRID.TBL:
    * Range templates:  STxxxxxx / SMxxxxxx / SWxxxxxx  (e.g. ST000010 = 0-10cm average)
    * Point-depth templates: SOILM### / SOILT###        (e.g. SOILM020 = value at 20cm)
  This prevents metgrid from creating SOILM/SOILT with zero vertical levels (which can crash ext_pkg_write_field).

Deps:
  pip install netCDF4 numpy pywinter
"""

from __future__ import annotations

import argparse
import bisect
import glob
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
from netCDF4 import Dataset

import pywinter.winter as pyw

# Physical constants used in thermodynamic conversions.
RD = 287.0  # Gas constant for dry air [J kg-1 K-1].
CP = 1004.0  # Specific heat of dry air at constant pressure [J kg-1 K-1].
P0 = 100000.0  # Reference pressure for potential temperature [Pa].
G = 9.81  # Gravitational acceleration [m s-2].
EPS = 1e-12  # Tiny epsilon to avoid divide-by-zero.

# WPS "surface" level identifier used by metgrid and pywinter.
WPS_SURFACE_LEVEL = "200100"  # IMPORTANT: must match METGRID.TBL "(200100)" and pywinter expects len(level)


# -----------------------------
# Minimal WPS namelist parser
# -----------------------------

@dataclass
class WpsConfig:
    """Container for the small subset of namelist.wps fields we care about."""
    max_dom: int = 1
    interval_seconds: int = 0
    ungrib_prefix: Optional[str] = None
    metgrid_fg_name: Optional[str] = None
    modlev_press_pa: List[float] = None
    geogrid_dx: Optional[float] = None
    geogrid_dy: Optional[float] = None
    parent_grid_ratio: List[int] = None

    def __post_init__(self):
        # Normalize optional lists to avoid None checks later.
        self.modlev_press_pa = self.modlev_press_pa or []
        self.parent_grid_ratio = self.parent_grid_ratio or []


def _strip_comments(line: str) -> str:
    """Remove Fortran-style comments and surrounding whitespace."""
    return line.split("!")[0].strip()


def _parse_string_value(s: str) -> str:
    """Extract a quoted string or fallback to a trimmed raw token."""
    m = re.search(r"['\"]([^'\"]+)['\"]", s)
    return m.group(1) if m else s.strip().strip(",").strip()


def _parse_int_value(rhs: str) -> int:
    """Parse the first integer found on the right-hand side of a namelist line."""
    m = re.search(r"[-+]?\d+", rhs)
    if not m:
        raise ValueError(f"Cannot parse int from: {rhs}")
    return int(m.group(0))


def _parse_float_value(rhs: str) -> float:
    """Parse the first float found on the right-hand side of a namelist line."""
    m = re.search(r"[-+]?\d+(?:\.\d+)?", rhs)
    if not m:
        raise ValueError(f"Cannot parse float from: {rhs}")
    return float(m.group(0))


def _parse_multiline_number_list(lines: List[str], start_idx: int) -> Tuple[List[float], int]:
    """Collect numeric values that may span multiple namelist lines."""
    vals: List[float] = []
    i = start_idx
    first = True
    while i < len(lines):
        raw = _strip_comments(lines[i])
        if not raw:
            i += 1
            continue
        if raw.startswith("/") or raw.startswith("&"):
            break
        if (not first) and re.match(r"^[A-Za-z_]\w*\s*=", raw):
            break

        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", raw)
        for n in nums:
            vals.append(float(n))

        first = False
        i += 1
    return vals, i


def parse_namelist_wps(path: str) -> WpsConfig:
    """Parse only the namelist.wps fields used by this converter."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    cfg = WpsConfig()  # Initialize with defaults.
    current_section: Optional[str] = None  # Track current &section.

    i = 0
    while i < len(lines):
        line = _strip_comments(lines[i])
        if not line:
            i += 1
            continue

        if line.startswith("&"):
            current_section = line[1:].strip().lower()
            i += 1
            continue

        if line.startswith("/"):
            current_section = None
            i += 1
            continue

        if "=" in line and current_section:
            key, rhs = [x.strip() for x in line.split("=", 1)]
            key_l = key.lower()

            if current_section == "share":
                if key_l == "max_dom":
                    cfg.max_dom = _parse_int_value(rhs)
                elif key_l == "interval_seconds":
                    cfg.interval_seconds = _parse_int_value(rhs)

            elif current_section == "ungrib":
                if key_l == "prefix":
                    cfg.ungrib_prefix = _parse_string_value(rhs)

            elif current_section == "metgrid":
                if key_l == "fg_name":
                    cfg.metgrid_fg_name = _parse_string_value(rhs)

            elif current_section == "mod_levs":
                if key_l == "press_pa":
                    vals, new_i = _parse_multiline_number_list(lines, i)
                    cfg.modlev_press_pa = vals
                    i = new_i
                    continue

            elif current_section == "geogrid":
                if key_l == "dx":
                    cfg.geogrid_dx = _parse_float_value(rhs)
                elif key_l == "dy":
                    cfg.geogrid_dy = _parse_float_value(rhs)
                elif key_l == "parent_grid_ratio":
                    vals, new_i = _parse_multiline_number_list(lines, i)
                    cfg.parent_grid_ratio = [int(round(v)) for v in vals]
                    i = new_i
                    continue

        i += 1

    return cfg


def parse_domain_token(s: str) -> int:
    """Convert a domain token like 'd04' or '4' into an integer."""
    s = s.strip().lower()
    if s.startswith("d"):
        s = s[1:]
    if not s.isdigit():
        raise argparse.ArgumentTypeError("Domain must be like d04 or 4")
    return int(s)


# -----------------------------
# METGRID.TBL parsing
# -----------------------------

def parse_metgrid_tbl_names(path: str) -> List[str]:
    """Extract ordered, unique field names from METGRID.TBL."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    names = re.findall(r"^\s*name\s*=\s*([A-Za-z0-9_]+)\s*$", txt, flags=re.MULTILINE)
    seen = set()
    out: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# Range templates
def is_soil_range_template(name: str) -> bool:
    """Check for soil range template names like ST000010."""
    return bool(re.fullmatch(r"(ST|SM|SW)\d{6}", name))


def parse_range_depth_cm(name: str) -> Tuple[int, int]:
    """Parse soil range template depth limits in centimeters."""
    m = re.fullmatch(r"(ST|SM|SW)(\d{3})(\d{3})", name)
    if not m:
        raise ValueError(f"Not a soil range template name: {name}")
    z0 = int(m.group(2))
    z1 = int(m.group(3))
    if z1 < z0:
        z0, z1 = z1, z0
    return z0, z1


# Point-depth templates
def is_soil_point_template(name: str) -> bool:
    """Check for soil point template names like SOILM020."""
    return bool(re.fullmatch(r"SOIL[MT]\d{3}", name))


def parse_point_depth_cm(name: str) -> int:
    """Parse a soil point depth in centimeters from its template name."""
    m = re.fullmatch(r"SOIL[MT](\d{3})", name)
    if not m:
        raise ValueError(f"Not a soil point template name: {name}")
    return int(m.group(1))


def build_wanted_set(tbl_names: List[str]) -> Set[str]:
    """
    We include:
      - common surface / pressure-level products
      - soil range templates (ST/SM/SWxxxxxx)
      - soil point templates (SOILM### / SOILT###) required by your METGRID.TBL
    """
    wanted: Set[str] = set()
    supported = {
        "PSFC", "LANDSEA", "SEAICE", "SNOW", "SST", "SKINTEMP",
        "CANWAT", "VEGCAT", "SOILCAT", "PMSL",
        "TT", "UU", "VV", "SPECHUMD", "GHT", "RH",
    }
    for n in tbl_names:
        if n in supported or is_soil_range_template(n) or is_soil_point_template(n):
            wanted.add(n)
    return wanted


# -----------------------------
# Interval inference
# -----------------------------

def infer_domain_dx_m(cfg: WpsConfig, dom: int) -> float:
    """Infer grid spacing for a nested domain using parent_grid_ratio."""
    if cfg.geogrid_dx is None:
        raise RuntimeError("Cannot infer dx: &geogrid dx missing in namelist.wps")
    if dom <= 1:
        return float(cfg.geogrid_dx)

    if not cfg.parent_grid_ratio or len(cfg.parent_grid_ratio) < dom:
        raise RuntimeError(
            f"Cannot infer dx: parent_grid_ratio missing/too short (len={len(cfg.parent_grid_ratio)}), dom={dom}"
        )

    ratio_prod = 1
    for d in range(2, dom + 1):
        ratio_prod *= int(cfg.parent_grid_ratio[d - 1])

    return float(cfg.geogrid_dx) / float(ratio_prod)


def round_to_60(x: float) -> int:
    return int(round(x / 60.0) * 60)


def best_interval_from_dx(dx_m: float) -> int:
    raw = round_to_60(3.0 * dx_m)
    return max(300, min(10800, raw))


# -----------------------------
# WRF helpers
# -----------------------------

def read_times(ds: Dataset) -> List[datetime]:
    """Read WRF 'Times' variable into a list of datetimes."""
    if "Times" not in ds.variables:
        raise RuntimeError("Missing Times variable in wrfout.")
    t = ds.variables["Times"][:]
    out: List[datetime] = []
    for row in t:
        s = row.tobytes().decode("ascii", errors="ignore").strip()
        out.append(datetime.strptime(s, "%Y-%m-%d_%H:%M:%S"))
    return out


def get_geo(ds: Dataset, iswin_grid: bool = True) -> object:
    """Build the pywinter georeference object from WRF metadata."""
    map_proj = int(getattr(ds, "MAP_PROJ"))
    dx_m = float(getattr(ds, "DX"))
    dy_m = float(getattr(ds, "DY"))
    truelat1 = float(getattr(ds, "TRUELAT1"))
    truelat2 = float(getattr(ds, "TRUELAT2"))
    stand_lon = float(getattr(ds, "STAND_LON"))

    if "XLAT" not in ds.variables or "XLONG" not in ds.variables:
        raise RuntimeError("Missing XLAT/XLONG in wrfout.")
    xlat = ds.variables["XLAT"][0, :, :]
    xlon = ds.variables["XLONG"][0, :, :]
    sw_lat = float(xlat[0, 0])
    sw_lon = float(xlon[0, 0])

    dx_km = dx_m / 1000.0
    dy_km = dy_m / 1000.0
    iswin = bool(iswin_grid)

    if map_proj == 1:
        return pyw.Geo3(sw_lat, sw_lon, dx_km, dy_km, stand_lon, truelat1, truelat2, iswin)
    if map_proj == 2:
        return pyw.Geo5(sw_lat, sw_lon, dx_km, dy_km, stand_lon, truelat1, iswin)
    if map_proj == 3:
        return pyw.Geo1(sw_lat, sw_lon, dx_km, dy_km, truelat1)
    if map_proj == 6:
        dlat = float(xlat[1, 0] - xlat[0, 0])
        dlon = float(xlon[0, 1] - xlon[0, 0])
        return pyw.Geo0(sw_lat, sw_lon, dlat, dlon)

    raise RuntimeError(f"Unsupported MAP_PROJ={map_proj}")


def vget(ds: Dataset, name: str, ti: int) -> Optional[np.ndarray]:
    """Return a 2D or 3D array from a WRF variable at time index ti."""
    if name not in ds.variables:
        return None
    v = ds.variables[name]
    if v.ndim == 3:
        return np.asarray(v[ti, :, :], dtype=np.float32)
    if v.ndim == 4:
        return np.asarray(v[ti, :, :, :], dtype=np.float32)
    return None


def destagger_x(u_stag: np.ndarray) -> np.ndarray:
    """Destagger U by averaging adjacent x-staggered grid cells."""
    return 0.5 * (u_stag[:, :, :-1] + u_stag[:, :, 1:])


def destagger_y(v_stag: np.ndarray) -> np.ndarray:
    """Destagger V by averaging adjacent y-staggered grid cells."""
    return 0.5 * (v_stag[:, :-1, :] + v_stag[:, 1:, :])


def temp_k_from_wrf(T_pert: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    """Convert WRF perturbation potential temperature to absolute temperature."""
    theta = T_pert + 300.0
    return theta * np.power(np.maximum(p_pa, 1.0) / P0, RD / CP)


def z_mass_from_ph(ph: np.ndarray, phb: np.ndarray) -> np.ndarray:
    """Compute mass-level geopotential height from staggered geopotential."""
    z_stag = (ph + phb) / G
    return 0.5 * (z_stag[:-1, :, :] + z_stag[1:, :, :])


def rotate_uv_to_earth(u: np.ndarray, v: np.ndarray, cosalpha: np.ndarray, sinalpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate grid-relative winds to earth-relative using rotation angles."""
    u_e = u * cosalpha - v * sinalpha
    v_e = u * sinalpha + v * cosalpha
    return u_e.astype(np.float32), v_e.astype(np.float32)


def interp_logp(var3d: np.ndarray, p3d: np.ndarray, plevs: np.ndarray) -> np.ndarray:
    """Interpolate a 3D field to pressure levels using log-pressure."""
    nz, ny, nx = var3d.shape
    nplev = plevs.size
    out = np.full((nplev, ny, nx), np.nan, dtype=np.float32)
    log_plev = np.log(np.maximum(plevs, 1.0))

    for j in range(ny):
        for i in range(nx):
            pcol = p3d[:, j, i]
            vcol = var3d[:, j, i]
            logp = np.log(np.maximum(pcol, 1.0))

            order = np.argsort(logp)
            logp_s = logp[order]
            v_s = vcol[order]

            ok = np.isfinite(logp_s) & np.isfinite(v_s)
            if np.count_nonzero(ok) < 2:
                continue

            out[:, j, i] = np.interp(log_plev, logp_s[ok], v_s[ok], left=np.nan, right=np.nan)

    return out


def dt_to_wps_stamp(dt: datetime, interval_seconds: int, force_hourly: bool) -> str:
    if force_hourly:
        return dt.strftime("%Y-%m-%d_%H")
    if interval_seconds and (interval_seconds % 3600 == 0) and (dt.minute == 0) and (dt.second == 0):
        return dt.strftime("%Y-%m-%d_%H")
    return dt.strftime("%Y-%m-%d_%H:%M")


# -----------------------------
# Derived fields (PMSL, RH)
# -----------------------------

def compute_pmsl(psfc: np.ndarray, hgt: np.ndarray, t2: np.ndarray, q2: np.ndarray) -> np.ndarray:
    tv = t2 * (1.0 + 0.61 * np.maximum(q2, 0.0))
    tv = np.maximum(tv, 150.0)
    return (psfc * np.exp(hgt / (RD * tv))).astype(np.float32)


def saturation_vapor_pressure_pa(Tk: np.ndarray) -> np.ndarray:
    Tc = Tk - 273.15
    es_hpa = 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5))
    return (es_hpa * 100.0).astype(np.float32)


def compute_rh_from_qv_T_p(qv: np.ndarray, Tk: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    eps = 0.622
    e = (qv * p_pa) / (eps + (1.0 - eps) * np.maximum(qv, 0.0) + EPS)
    es = saturation_vapor_pressure_pa(Tk)
    rh = 100.0 * (e / (es + EPS))
    return np.clip(rh, 0.0, 100.0).astype(np.float32)


# -----------------------------
# Soil computations
# -----------------------------

def soil_interfaces_from_dzs(ds: Dataset, ti: int) -> Optional[np.ndarray]:
    """Interfaces in meters from DZS thicknesses."""
    if "DZS" not in ds.variables:
        return None
    dzs = np.asarray(ds.variables["DZS"][ti, :], dtype=np.float32).reshape(-1)
    if np.any(~np.isfinite(dzs)) or np.any(dzs <= 0):
        return None
    zint = np.zeros((dzs.size + 1,), dtype=np.float32)
    zint[1:] = np.cumsum(dzs)
    return zint


def soil_centers_from_zs(ds: Dataset, ti: int) -> Optional[np.ndarray]:
    """Centers in meters from ZS (depths of centers of soil layers)."""
    if "ZS" not in ds.variables:
        return None
    zs = np.asarray(ds.variables["ZS"][ti, :], dtype=np.float32).reshape(-1)
    if np.any(~np.isfinite(zs)) or np.any(zs < 0):
        return None
    return zs


def depth_average_from_layers(var_layers: np.ndarray, zint_m: np.ndarray, z0_m: float, z1_m: float) -> np.ndarray:
    """Average over [z0,z1] using layer slabs defined by zint_m."""
    nsoil = var_layers.shape[0]
    z0 = float(min(z0_m, z1_m))
    z1 = float(max(z0_m, z1_m))
    if z1 <= z0:
        return np.full(var_layers.shape[1:], np.nan, dtype=np.float32)

    ny, nx = var_layers.shape[1], var_layers.shape[2]
    num = np.zeros((ny, nx), dtype=np.float64)
    den = np.zeros((ny, nx), dtype=np.float64)

    for k in range(nsoil):
        top = float(zint_m[k])
        bot = float(zint_m[k + 1])
        ov = max(0.0, min(bot, z1) - max(top, z0))
        if ov <= 0.0:
            continue
        slab = var_layers[k, :, :].astype(np.float64)
        mask = np.isfinite(slab)
        num[mask] += slab[mask] * ov
        den[mask] += ov

    out = np.full((ny, nx), np.nan, dtype=np.float32)
    ok = den > 0
    out[ok] = (num[ok] / den[ok]).astype(np.float32)
    return out


def interp_to_depth(var_layers: np.ndarray, zc_m: np.ndarray, target_m: float) -> np.ndarray:
    """
    Interpolate layer-center values to a given depth (meters).
    Per-gridpoint interpolation is not needed because zc is 1D (same for all gridpoints).
    """
    zc = np.asarray(zc_m, dtype=np.float32).reshape(-1)
    if zc.size != var_layers.shape[0]:
        return None

    # Ensure monotonic
    order = np.argsort(zc)
    zc = zc[order]
    v = var_layers[order, :, :]

    # Clamp outside range
    if target_m <= float(zc[0]):
        return v[0, :, :].astype(np.float32)
    if target_m >= float(zc[-1]):
        return v[-1, :, :].astype(np.float32)

    # Find bracketing centers
    k = int(np.searchsorted(zc, target_m))
    z0, z1 = float(zc[k - 1]), float(zc[k])
    w = (target_m - z0) / (z1 - z0 + EPS)

    return ((1.0 - w) * v[k - 1, :, :] + w * v[k, :, :]).astype(np.float32)


def compute_soil_range_template(ds: Dataset, ti: int, name: str) -> Optional[np.ndarray]:
    """ST/SM/SWxxxxxx => depth-average over [z0,z1] cm."""
    if not is_soil_range_template(name):
        return None
    z0_cm, z1_cm = parse_range_depth_cm(name)
    z0_m, z1_m = z0_cm / 100.0, z1_cm / 100.0

    prefix = name[:2]
    src = {"ST": "TSLB", "SM": "SMOIS", "SW": "SH2O"}.get(prefix)
    if not src or src not in ds.variables:
        return None

    zint = soil_interfaces_from_dzs(ds, ti)
    if zint is None:
        return None

    var = np.asarray(ds.variables[src][ti, :, :, :], dtype=np.float32)
    if var.ndim != 3:
        return None

    return depth_average_from_layers(var, zint, z0_m, z1_m)


def compute_soil_point_template(ds: Dataset, ti: int, name: str) -> Optional[np.ndarray]:
    """
    SOILM### / SOILT### => interpolate to depth ### cm using ZS (layer centers).
    This matches how METGRID.TBL expects SOILM000, SOILM005, ... at level (200100).
    """
    if not is_soil_point_template(name):
        return None

    depth_cm = parse_point_depth_cm(name)
    target_m = depth_cm / 100.0

    if name.startswith("SOILM"):
        src = "SMOIS"
    else:
        src = "TSLB"

    if src not in ds.variables:
        return None

    zc = soil_centers_from_zs(ds, ti)
    if zc is None:
        return None

    var = np.asarray(ds.variables[src][ti, :, :, :], dtype=np.float32)
    if var.ndim != 3:
        return None

    return interp_to_depth(var, zc, target_m)


# -----------------------------
# Extraction / interpolation / writing
# -----------------------------

def extract_fields_as_arrays(
    ds: Dataset,
    ti: int,
    plevs: Optional[np.ndarray],
    rotate_to_earth: bool,
    wanted: Optional[Set[str]] = None
) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray], str]]:
    out: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], str]] = {}

    mapping_2d = [
        ("PSFC", "PSFC"),
        ("LANDSEA", "LANDMASK"),
        ("SEAICE", "SEAICE"),
        ("SNOW", "SNOW"),
        ("SST", "SST"),
        ("SKINTEMP", "TSK"),
        ("CANWAT", "CANWAT"),
        ("VEGCAT", "IVGTYP"),
        ("SOILCAT", "ISLTYP"),
    ]
    for outname, wrfvar in mapping_2d:
        if wanted is not None and outname not in wanted:
            continue
        arr = vget(ds, wrfvar, ti)
        if arr is not None:
            out[outname] = (arr.astype(np.float32), None, "2d")

    if wanted is None or "PMSL" in wanted:
        psfc = vget(ds, "PSFC", ti)
        hgt = vget(ds, "HGT", ti)
        t2 = vget(ds, "T2", ti)
        q2 = vget(ds, "Q2", ti)
        if psfc is not None and hgt is not None and t2 is not None and q2 is not None:
            out["PMSL"] = (compute_pmsl(psfc, hgt, t2, q2), None, "2d")

    # Soil templates requested by METGRID.TBL
    if wanted is not None:
        # Range templates (ST/SM/SWxxxxxx)
        for n in [x for x in wanted if is_soil_range_template(x)]:
            val = compute_soil_range_template(ds, ti, n)
            if val is not None:
                out[n] = (val.astype(np.float32), None, "2d")

        # Point templates (SOILM### / SOILT###)
        for n in [x for x in wanted if is_soil_point_template(x)]:
            val = compute_soil_point_template(ds, ti, n)
            if val is not None:
                out[n] = (val.astype(np.float32), None, "2d")

    if plevs is None or plevs.size == 0:
        return out

    need_any_3d = (wanted is None) or any(x in wanted for x in ["TT", "UU", "VV", "SPECHUMD", "GHT", "RH"])
    if not need_any_3d:
        return out

    T_pert = vget(ds, "T", ti)
    P = vget(ds, "P", ti)
    PB = vget(ds, "PB", ti)
    QV = vget(ds, "QVAPOR", ti)
    U_stag = vget(ds, "U", ti)
    V_stag = vget(ds, "V", ti)
    PH = vget(ds, "PH", ti)
    PHB = vget(ds, "PHB", ti)

    if any(x is None for x in [T_pert, P, PB, QV, U_stag, V_stag, PH, PHB]):
        return out

    p_full = (P + PB).astype(np.float32)
    t_k = temp_k_from_wrf(T_pert, p_full).astype(np.float32)
    z_m = z_mass_from_ph(PH, PHB).astype(np.float32)
    u = destagger_x(U_stag).astype(np.float32)
    v = destagger_y(V_stag).astype(np.float32)

    if rotate_to_earth and ("COSALPHA" in ds.variables) and ("SINALPHA" in ds.variables):
        ca = vget(ds, "COSALPHA", ti)
        sa = vget(ds, "SINALPHA", ti)
        if ca is not None and sa is not None:
            u, v = rotate_uv_to_earth(u, v, ca, sa)

    def want(n: str) -> bool:
        return wanted is None or (n in wanted)

    if want("TT"):
        out["TT_3D"] = (interp_logp(t_k, p_full, plevs), plevs, "3dp")
    if want("UU"):
        out["UU_3D"] = (interp_logp(u, p_full, plevs), plevs, "3dp")
    if want("VV"):
        out["VV_3D"] = (interp_logp(v, p_full, plevs), plevs, "3dp")
    if want("SPECHUMD"):
        out["SPECHUMD_3D"] = (interp_logp(QV.astype(np.float32), p_full, plevs), plevs, "3dp")
    if want("GHT"):
        out["GHT_3D"] = (interp_logp(z_m, p_full, plevs), plevs, "3dp")
    if want("RH"):
        rh3d = compute_rh_from_qv_T_p(QV.astype(np.float32), t_k, p_full)
        out["RH_3D"] = (interp_logp(rh3d, p_full, plevs), plevs, "3dp")

    return out


def interpolate_field_dict(
    f0: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], str]],
    f1: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], str]],
    alpha: float
) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray], str]]:
    out: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], str]] = {}
    common = set(f0.keys()).intersection(f1.keys())
    for k in common:
        a0, lev0, kind0 = f0[k]
        a1, lev1, kind1 = f1[k]
        if kind0 != kind1:
            continue
        if (lev0 is None) != (lev1 is None):
            continue
        if lev0 is not None and lev1 is not None:
            if lev0.shape != lev1.shape or not np.allclose(lev0, lev1):
                continue

        m0 = np.isfinite(a0)
        m1 = np.isfinite(a1)

        out_arr = np.empty_like(a0, dtype=np.float32)
        both = m0 & m1
        only0 = m0 & ~m1
        only1 = ~m0 & m1
        neither = ~m0 & ~m1

        out_arr[both] = (1.0 - alpha) * a0[both] + alpha * a1[both]
        out_arr[only0] = a0[only0]
        out_arr[only1] = a1[only1]
        out_arr[neither] = np.nan

        out[k] = (out_arr, lev0, kind0)
    return out


# -----------------------------
# pywinter object creation (robust)
# -----------------------------

def v2d(name: str, arr: np.ndarray, desc: str, units: str, level: str) -> object:
    # IMPORTANT: level must be a string and should be "200100" for surface/soil fields
    return pyw.V2d(name, arr, desc, units, str(level))


def v2d_with_metadata(name: str, arr: np.ndarray) -> object:
    """
    Always explicit metadata to avoid pywinter registry issues.
    Always level="200100" for our 2D fields so metgrid can match "(200100)".
    """
    # Soil range templates
    if is_soil_range_template(name):
        z0, z1 = parse_range_depth_cm(name)
        if name.startswith("ST"):
            return v2d(name, arr, f"Soil temperature avg {z0}-{z1} cm", "K", WPS_SURFACE_LEVEL)
        if name.startswith("SM"):
            return v2d(name, arr, f"Soil moisture avg {z0}-{z1} cm", "m3 m-3", WPS_SURFACE_LEVEL)
        return v2d(name, arr, f"Soil liquid water avg {z0}-{z1} cm", "m3 m-3", WPS_SURFACE_LEVEL)

    # Soil point templates
    if is_soil_point_template(name):
        d = parse_point_depth_cm(name)
        if name.startswith("SOILM"):
            return v2d(name, arr, f"Soil moisture at {d} cm", "m3 m-3", WPS_SURFACE_LEVEL)
        return v2d(name, arr, f"Soil temperature at {d} cm", "K", WPS_SURFACE_LEVEL)

    meta = {
        "PSFC": ("Surface pressure", "Pa"),
        "PMSL": ("Mean sea level pressure (approx)", "Pa"),
        "LANDSEA": ("Land/sea mask (1=land, 0=water)", "flag"),
        "SEAICE": ("Sea ice flag", "flag"),
        "SNOW": ("Snow water equivalent", "kg m-2"),
        "SST": ("Sea surface temperature", "K"),
        "SKINTEMP": ("Skin temperature", "K"),
        "CANWAT": ("Canopy water", "kg m-2"),
        "VEGCAT": ("Dominant vegetation category", "category"),
        "SOILCAT": ("Dominant soil category", "category"),
    }
    desc, units = meta.get(name, (f"{name} (generated)", "unknown"))
    return v2d(name, arr, desc, units, WPS_SURFACE_LEVEL)


def to_pywinter_fields(fd: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], str]]) -> List[object]:
    out: List[object] = []

    # 2D first
    keys_2d = [k for k, v in fd.items() if v[2] == "2d"]
    for k in sorted(keys_2d):
        arr, _, _ = fd[k]
        out.append(v2d_with_metadata(k, arr))

    # 3D pressure levels
    mapping = {
        "TT_3D": "TT",
        "UU_3D": "UU",
        "VV_3D": "VV",
        "SPECHUMD_3D": "SPECHUMD",
        "GHT_3D": "GHT",
        "RH_3D": "RH",
    }
    for k, wn in mapping.items():
        if k in fd:
            arr, plevs, _ = fd[k]
            out.append(pyw.V3dp(wn, arr, plevs))

    return out


# -----------------------------
# I/O utilities
# -----------------------------

def expand_inputs(items: List[str]) -> List[str]:
    files: List[str] = []
    for item in items:
        m = glob.glob(item)
        if m:
            files.extend(m)
        elif os.path.exists(item):
            files.append(item)
    return sorted(dict.fromkeys(files))


def generate_target_times(start: datetime, end: datetime, step_s: int) -> List[datetime]:
    out: List[datetime] = []
    t = start
    step = timedelta(seconds=step_s)
    while t <= end:
        out.append(t)
        t += step
    return out


def infer_wrfout_interval_seconds(available_times: List[datetime]) -> int:
    if len(available_times) < 2:
        return 0
    deltas = [(available_times[i + 1] - available_times[i]).total_seconds() for i in range(len(available_times) - 1)]
    return int(round(float(np.median(deltas))))


def build_time_index(files: List[str], iswin_grid: bool) -> Tuple[object, List[datetime], Dict[datetime, Tuple[str, int]]]:
    geo = None
    idx: Dict[datetime, Tuple[str, int]] = {}

    for fp in files:
        with Dataset(fp, "r") as ds:
            if geo is None:
                geo = get_geo(ds, iswin_grid=iswin_grid)
            times = read_times(ds)
            for ti, dt in enumerate(times):
                idx[dt] = (fp, ti)

    if geo is None:
        raise RuntimeError("No readable wrfout files found.")
    available = sorted(idx.keys())
    return geo, available, idx


# -----------------------------
# main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Convert wrfout to WPS intermediate files for metgrid.exe")
    ap.add_argument("-i", "--input", required=True, nargs="+",
                    help="Input wrfout path(s) and/or glob(s). Quote globs if desired.")
    ap.add_argument("-o", "--outdir", required=True, help="Output directory for intermediate files.")

    ap.add_argument("--namelist", default="", help="Path to namelist.wps to infer prefix/plevs/interval_seconds/max_dom/dx.")
    ap.add_argument("--domain", type=parse_domain_token, default=None,
                    help="Target domain (e.g., d04 or 4). Defaults to max_dom when --namelist is provided.")
    ap.add_argument("--infer-interval", action="store_true",
                    help="Infer best interval_seconds from inferred domain dx (dx + parent_grid_ratio chain).")
    ap.add_argument("--wrfout-interval", type=int, default=0,
                    help="Override wrfout cadence in seconds. If 0, inferred from wrfout Times.")
    ap.add_argument("--metgrid-table", default="",
                    help="Optional path to METGRID.TBL. If provided, generate variables requested by the table when possible.")

    ap.add_argument("--prefix", default=None,
                    help="Intermediate prefix override (else inferred from namelist &ungrib prefix, else FILE).")
    ap.add_argument("--force-hourly-stamp", action="store_true",
                    help="Force filenames as YYYY-MM-DD_HH (no minutes).")
    ap.add_argument("--plevs-pa", default="",
                    help="Comma-separated pressure levels in Pa for 3D fields. If omitted, inferred from namelist &mod_levs press_pa.")
    ap.add_argument("--rotate-winds-to-earth", action="store_true",
                    help="Rotate 3D U/V to earth-relative using COSALPHA/SINALPHA (if present).")
    ap.add_argument("--iswin", choices=["grid", "earth"], default="grid",
                    help="pywinter geoinfo 'iswin' flag. Use 'grid' if winds are grid-relative (default).")

    args = ap.parse_args()

    cfg: Optional[WpsConfig] = None
    target_dom: Optional[int] = args.domain
    namelist_interval = 0
    inferred_prefix: Optional[str] = None
    inferred_plevs: List[float] = []

    if args.namelist:
        cfg = parse_namelist_wps(args.namelist)
        namelist_interval = cfg.interval_seconds or 0
        inferred_prefix = cfg.ungrib_prefix
        inferred_plevs = cfg.modlev_press_pa or []
        if target_dom is None:
            target_dom = cfg.max_dom
        if target_dom > cfg.max_dom:
            raise SystemExit(f"--domain d{target_dom:02d} exceeds max_dom={cfg.max_dom} in {args.namelist}")

    prefix = args.prefix or inferred_prefix or "FILE"

    if args.plevs_pa.strip():
        plevs = np.array([float(x.strip()) for x in args.plevs_pa.split(",") if x.strip()], dtype=np.float32)
    else:
        plevs = np.array(inferred_plevs, dtype=np.float32) if inferred_plevs else None

    wanted: Optional[Set[str]] = None
    if args.metgrid_table:
        tbl_names = parse_metgrid_tbl_names(args.metgrid_table)
        wanted = build_wanted_set(tbl_names)

        soil_range = sum(1 for n in wanted if is_soil_range_template(n))
        soil_point = sum(1 for n in wanted if is_soil_point_template(n))

        print("== METGRID.TBL ==")
        print(f"  table                 : {args.metgrid_table}")
        print(f"  names in table        : {len(tbl_names)}")
        print(f"  supported by converter: {len(wanted)}")
        print(f"  soil range templates  : {soil_range}")
        print(f"  soil point templates  : {soil_point}")
        demo = sorted([n for n in wanted if is_soil_point_template(n)])[:10]
        if demo:
            print(f"  soil point sample     : {', '.join(demo)}{' ...' if soil_point > 10 else ''}")
        print("=================")

    os.makedirs(args.outdir, exist_ok=True)
    files = expand_inputs(args.input)
    if not files:
        raise SystemExit(f"No inputs matched: {args.input}")

    geo, available_times, time_index = build_time_index(files, iswin_grid=(args.iswin == "grid"))

    wrfout_interval = args.wrfout_interval or infer_wrfout_interval_seconds(available_times) or 0
    if wrfout_interval <= 0:
        wrfout_interval = 3600

    output_interval = namelist_interval if namelist_interval > 0 else wrfout_interval
    inferred_dx = None
    if cfg and args.infer_interval:
        inferred_dx = infer_domain_dx_m(cfg, target_dom or cfg.max_dom)
        output_interval = best_interval_from_dx(inferred_dx)

    if cfg:
        print("== From namelist.wps ==")
        print(f"  max_dom           : {cfg.max_dom}")
        print(f"  target domain     : d{(target_dom or cfg.max_dom):02d}")
        print(f"  namelist interval : {namelist_interval}")
        print(f"  prefix            : {prefix}")
        if cfg.parent_grid_ratio:
            print(f"  parent_grid_ratio : {cfg.parent_grid_ratio[:max(cfg.max_dom, 1)]}")
        if inferred_dx is not None:
            print(f"  inferred dx       : {inferred_dx:.1f} m")
        print("======================")

    print("== Time handling ==")
    print(f"  wrfout interval (detected) : {wrfout_interval} s")
    print(f"  output interval            : {output_interval} s")
    print("  mode                       : " +
          ("densify via interpolation" if output_interval < wrfout_interval else "direct / no densification"))
    print("===================")

    start = available_times[0]
    end = available_times[-1]
    target_times = generate_target_times(start, end, output_interval)
    avail_list = available_times

    for t in target_times:
        if t in time_index:
            fp, ti = time_index[t]
            with Dataset(fp, "r") as ds:
                fd = extract_fields_as_arrays(
                    ds, ti, plevs, rotate_to_earth=args.rotate_winds_to_earth, wanted=wanted
                )
        else:
            pos = bisect.bisect_left(avail_list, t)
            if pos == 0 or pos == len(avail_list):
                continue
            t0 = avail_list[pos - 1]
            t1 = avail_list[pos]
            fp0, ti0 = time_index[t0]
            fp1, ti1 = time_index[t1]
            denom = (t1 - t0).total_seconds()
            if denom <= 0:
                continue
            alpha = (t - t0).total_seconds() / denom

            with Dataset(fp0, "r") as ds0, Dataset(fp1, "r") as ds1:
                f0 = extract_fields_as_arrays(
                    ds0, ti0, plevs, rotate_to_earth=args.rotate_winds_to_earth, wanted=wanted
                )
                f1 = extract_fields_as_arrays(
                    ds1, ti1, plevs, rotate_to_earth=args.rotate_winds_to_earth, wanted=wanted
                )
                fd = interpolate_field_dict(f0, f1, float(alpha))

        stamp = dt_to_wps_stamp(t, output_interval, args.force_hourly_stamp)
        fields = to_pywinter_fields(fd)

        pyw.cinter(prefix, stamp, geo, fields, args.outdir)
        print(f"Wrote {prefix}:{stamp} ({len(fields)} fields)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
