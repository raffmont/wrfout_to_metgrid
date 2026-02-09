# wrfout_to_metgrid

Create WPS intermediate files from WRF `wrfout` NetCDF history files so that `metgrid.exe` can ingest them.

## Requirements

- Python 3
- `netCDF4`, `numpy`, `pywinter`

```bash
pip install netCDF4 numpy pywinter
```

## Basic usage

```bash
./wrfout_to_wpsinit.py \
  --input wrfout_d01_* \
  --outdir ./metgrid_inputs \
  --namelist namelist.wps \
  --metgrid-table METGRID.TBL
```

## Notes

- If `--namelist` is omitted but `namelist.wps` exists in the current directory, the script will use it automatically
  to keep interval stamps and pressure levels aligned with `metgrid.exe`.
- Ensure `&mod_levs press_pa` is set in `namelist.wps` (or pass `--plevs-pa`) so mandatory 3D fields like `TT` can
  be written for `metgrid.exe`.
- Use `--infer-interval` to derive output cadence from the domain grid spacing.
- Soil fields are generated to match soil range (`ST/SM/SWxxxxxx`) and point-depth (`SOILM###/SOILT###`) templates present in `METGRID.TBL`.
