# CVDP Metrics Summary

NCL version metrics from [CVDP-ncl](https://github.com/NCAR/CVDP-ncl), organized by category. Each entry notes whether a Python implementation exists in this package.

---

## Climatological Statistics (Mean & Standard Deviation)

Seasonal (DJF, JFM, MAM, JJA, JAS, SON, ANN) climatological means and standard deviations computed without removing the annual cycle.

| Metric | Variable | Python Implementation |
|--------|----------|-----------------------|
| SST Mean & Std Dev | TS / ts | `cvdp/diag/climatology.py:compute_seasonal_avgs()`, `cvdp/diag/AtmOcnMean.py:mean_seasonal_calc()` |
| TAS Mean & Std Dev | TREFHT / tas | same as above |
| PSL Mean & Std Dev | PSL / psl | same as above (uses 5-month smoothing for NDJFM) |
| PR Mean & Std Dev | PRECT / pr | same as above |
| SICONC Mean & Std Dev | aice_nh, aice_sh / siconc | same as above (variable support TBD) |
| ZOS Mean & Std Dev | SSH / zos | same as above (variable support TBD) |

---

## Trends & Timeseries

Linear trends and global/regional area-averaged timeseries, computed per season.

| Metric | Variable | Python Implementation |
|--------|----------|-----------------------|
| SST Trends & Timeseries | TS / ts | `cvdp/diag/linear_trends.py:lin_regress()`, `cvdp/cvdp_utils/avg_functions.py:seasonal_trends_timeseries()`, `cvdp/vis/AtmOcnGR.py:compute_trend()` |
| TAS Trends & Timeseries | TREFHT / tas | same as above |
| PSL Trends & Timeseries | PSL / psl | same as above |
| PR Trends & Timeseries | PRECT / pr | same as above |
| SICONC Trends & Timeseries | siconc | same as above (variable support TBD) |
| ZOS Trends | SSH / zos | same as above (variable support TBD) |
| Global area-averaged timeseries | all variables | `cvdp/diag/time_series.py:seasonal_timeseries()`, `cvdp/cvdp_utils/avg_functions.py:seasonal_timeseries()` |

---

## ENSO (El Niño-Southern Oscillation)

All metrics derive from the Niño3.4 index: area-averaged SST anomalies over 5°S–5°N, 120–170°W. El Niño/La Niña events defined when the December Niño3.4 value (3-point binomial smoothed) exceeds ±1 std dev.

| Metric | Description | Python Implementation |
|--------|-------------|-----------------------|
| Niño3.4 index timeseries | Monthly SST anomaly index | **Not implemented** |
| Niño3.4 monthly std devs | Std dev by calendar month | **Not implemented** |
| Niño3.4 autocorrelations | Monthly autocorrelation function | **Not implemented** |
| Niño3.4 power spectra | Frequency-domain variance | **Not implemented** |
| Niño3.4 wavelet | Morlet wavelet (wavenumber=6), 95% chi-square significance | **Not implemented** |
| ENSO spatial composites | Composite maps of SST, PSL, PR, TAS during El Niño / La Niña | **Not implemented** |
| El Niño / La Niña Hovmöllers | 3°S–3°N meridional mean SST, Jan yr0–May yr+2 | **Not implemented** |
| SST indices | Niño 1+2, 3, 3.4, 4; Tropical N/S Atlantic; Tropical Indian Ocean | **Not implemented** |

---

## Pacific Decadal Modes

| Metric | Definition | Python Implementation |
|--------|------------|-----------------------|
| **PDO / PDV** | Leading PC of North Pacific (20–70°N, 110°E–100°W) area-weighted SST anomalies; 61-month low-pass filter applied | **Not implemented** (EOF infrastructure exists: `cvdp/diag/eof.py:get_eof()`, `cvdp/vis/AtmOcnGR.py:compute_eof()`) |
| **IPV (EOF-based)** | Leading PC of 13-yr low-pass filtered Pacific (40°S–60°N, 110°E–70°W) SST anomalies; requires ≥40 yrs | **Not implemented** |
| **IPV-Henley** | Central tropical Pacific (10°S–10°N, 170°E–90°W) SST anomalies minus average of NW and SW Pacific boxes | **Not implemented** |

---

## Atlantic Modes

| Metric | Definition | Python Implementation |
|--------|------------|-----------------------|
| **AMV** | Area-weighted SST anomalies over North Atlantic (0–60°N, 80°W–0°E); 61-month low-pass filter; regression maps also computed with 10-yr running mean ("Regr LP") | **Not implemented** |

---

## Atmospheric Modes of Variability (PSL-based EOFs)

All modes use seasonal/annual PSL averages with √cos(lat) weighting applied before EOF decomposition. Patterns are formed by regressing global PSL anomalies onto the normalized PC timeseries.

| Metric | Domain | EOF # | Python Implementation |
|--------|--------|-------|-----------------------|
| **NAM** (Northern Annular Mode) | 20–90°N | EOF 1 | `cvdp/diag/eof.py:get_eof()`, `cvdp/vis/AtmOcnGR.py:compute_eof()` — domain/wiring not finalized |
| **NAO** (North Atlantic Oscillation) | 20–80°N, 90°W–40°E | EOF 1 | same — domain/wiring not finalized |
| **PNA** (Pacific-North American) | 20–85°N, 120°E–120°W | EOF 1 | same |
| **NPO** (North Pacific Oscillation) | 20–85°N, 120°E–120°W | EOF 2 | same |
| **SAM** (Southern Annular Mode) | 20–90°S | EOF 1 | same |
| **PSA1** (Pacific-South American 1) | 20–90°S | EOF 2 | same |
| **PSA2** (Pacific-South American 2) | 20–90°S | EOF 3 | same |
| **SOI** (Southern Oscillation Index) | PSL difference: 30°S–0°N, 70–170°E minus 30°S–0°N, 160–80°W | — | **Not implemented** (`compute_npi()` in `cvdp/vis/AtmOcnGR.py` is an NPI analog, not SOI) |

---

## Ocean Circulation

| Metric | Variable | Description | Python Implementation |
|--------|----------|-------------|-----------------------|
| **AMOC** | MOC / msftmz | Maximum of the zonally-integrated, vertically-accumulated Atlantic meridional streamfunction (in Sv) | **Not implemented** |

---

## Regional Timeseries

A set of basin- and region-averaged indices computed from monthly fields of SST, PSL, PR, and TAS. See [NCL methodology page](https://webext.cgd.ucar.edu/Multi-Case/CVDP_repository/cesm2-lens_quadquad_1850-2100/methodology.html) for the full index list.

**Python implementation:** **Not implemented**

---

## Summary Table

| Category | Total Metrics | Implemented | Not Yet Implemented |
|----------|--------------|-------------|---------------------|
| Climatological Mean & Std Dev | 6 | 6 | 0 |
| Trends & Timeseries | 7 | 7 | 0 |
| ENSO | 8 | 0 | 8 |
| Pacific Decadal Modes | 3 | 0 | 3 |
| Atlantic Modes (AMV) | 1 | 0 | 1 |
| Atmospheric EOF Modes | 8 | partial (infrastructure only) | 8 |
| Ocean Circulation (AMOC) | 1 | 0 | 1 |
| Regional Timeseries | 1 | 0 | 1 |
| **Total** | **35** | **13** | **22** |

> "Implemented" counts metrics where end-to-end Python code exists. EOF infrastructure (`get_eof`, `compute_eof`) is present but the per-mode domain definitions and output pipeline are not finalized.

---

## Input Variables

| CESM Name | CMIP6 Name | Description | Frequency |
|-----------|-----------|-------------|-----------|
| TS | ts | Sea Surface / Land Surface Temperature | Monthly |
| TREFHT | tas | 2-m Air Temperature | Monthly |
| PSL | psl | Sea Level Pressure | Monthly |
| PRECT (PRECC+PRECL) | pr | Total Precipitation | Monthly |
| SSH | zos | Sea Surface Height | Monthly |
| aice_nh / aice_sh | siconc | Sea Ice Concentration | Monthly |
| MOC | msftmz | Ocean Meridional Overturning Streamfunction | Monthly |
