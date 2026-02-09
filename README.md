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

- Use `--infer-interval` to derive output cadence from the domain grid spacing.
- Soil fields are generated to match soil range (`ST/SM/SWxxxxxx`) and point-depth (`SOILM###/SOILT###`) templates present in `METGRID.TBL`.
