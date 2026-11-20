# CVDP Metrics Summary

Metrics from [CVDP-ncl](https://github.com/NCAR/CVDP-ncl) (v6.1.0), organized by category, with the Python implementation in `cvdp/metrics/`. Definitions follow the CVDP-ncl source scripts. Where these differ from older CVDP documentation, the source wins.

All functions are generic: one function serves every variable. Bounds are given as latitude south–north, longitude west–east.

---

## Climatological Statistics (Mean & Standard Deviation)

Seasonal (DJF, JFM, MAM, JJA, JAS, SON, ANN) climatological means and standard deviations computed without removing the annual cycle.

| Metric | Variable | Python Implementation |
|--------|----------|-----------------------|
| SST Mean & Std Dev | TS / ts | `climatology.get_seasonal_statistics()` |
| TAS Mean & Std Dev | TREFHT / tas | same as above |
| PSL Mean & Std Dev | PSL / psl | same as above |
| PR Mean & Std Dev | PRECT / pr | same as above |
| SICONC Mean & Std Dev | aice_nh, aice_sh / siconc | same as above; hemispheric sea-ice extent series in `sea_ice.sea_ice_extent()` |
| ZOS Mean & Std Dev | SSH / zos | same as above |

---

## Trends & Timeseries

Linear trends and global area-averaged timeseries, computed per season.

| Metric | Variable | Python Implementation |
|--------|----------|-----------------------|
| SST / TAS / PSL / PR / SICONC / ZOS Trends | all | `trends.trend_maps()` (change per decade) |
| Global area-averaged timeseries | all | `trends.seasonal_timeseries()` |

---

## ENSO (El Niño-Southern Oscillation)

All metrics derive from the Niño3.4 index: area-averaged SST anomalies over 5°S–5°N, 120–170°W. El Niño/La Niña events are defined when the December Niño3.4 value (3-point binomial smoothed) exceeds ±1 std dev.

| Metric | Description | Python Implementation |
|--------|-------------|-----------------------|
| Niño3.4 index timeseries | Monthly SST anomaly index | `enso.nino34_index()` |
| Niño3.4 monthly std devs | Std dev by calendar month | `enso.nino34_monthly_stddev()` |
| Niño3.4 autocorrelations | Monthly autocorrelation function | `enso.nino34_autocorrelation()` |
| SST indices | Niño 1+2, 3, 3.4, 4; Tropical N/S Atlantic; Tropical Indian Ocean; North Atlantic; Atlantic Niño; Southern Ocean; Atlantic Meridional Mode; Indian Ocean Dipole | `enso.sst_indices()` |
| Niño3.4 power spectra | Smoothed spectrum with Markov red-noise spectrum and 95%/99% confidence bounds | `enso.power_spectrum()` (also used for the PDO, IPV and AMV spectra) |
| Niño3.4 wavelet | Morlet wavelet (wavenumber=6), 95% red-noise significance, cone of influence | `enso.nino34_wavelet()` |
| ENSO spatial composites | Seasonal (JJA0, SON0, DJF+1, MAM+1) composites of SST, PSL, PR, TAS for El Niño, La Niña, and their difference | `enso.enso_composites()` |
| El Niño / La Niña Hovmöllers | 3°S–3°N mean SST anomalies, 120°E–80°W, Jan yr0–May yr+2 | `enso.enso_hovmoller()` |

---

## Pacific Decadal Modes

| Metric | Definition | Python Implementation |
|--------|------------|-----------------------|
| **PDO / PDV** | Leading PC of monthly North Pacific (20–70°N, 110°E–100°W) area-weighted SST anomalies; pattern = SST anomalies regressed onto the standardized PC | `modes.pdo()` |
| **IPV (EOF-based)** | Leading PC of 13-yr low-pass filtered (Lanczos) Pacific (40°S–60°N, 110°E–70°W) SST anomalies; requires ≥40 yrs | `modes.ipv_eof()` |
| **IPV-Henley** | Central tropical Pacific (10°S–10°N, 170°E–90°W) SST anomalies minus the average of the North (25–45°N, 140°E–145°W) and South (15–50°S, 150°E–160°W) Pacific boxes | `modes.ipv_henley()` |

---

## Atlantic Modes

| Metric | Definition | Python Implementation |
|--------|------------|-----------------------|
| **AMV** | Area-weighted SST anomalies over the North Atlantic (0–60°N, 80°W–0°E); 121-month running-mean low-pass index; regression maps smoothed with a 9-point filter; low-pass maps ("Regr LP") use the 121-month running mean on both index and field | `modes.amv()` |

---

## Atmospheric Modes of Variability (PSL-based)

EOF modes use seasonal/annual PSL averages with √cos(lat) weighting applied before EOF decomposition. Patterns are formed by regressing global PSL anomalies onto the normalized PC timeseries. All seven EOF modes share `modes.seasonal_eof_mode()`.

| Metric | Domain | EOF # | Python Implementation |
|--------|--------|-------|-----------------------|
| **NAM** (Northern Annular Mode) | 20–90°N | EOF 1 | `modes.nam()` |
| **NAO** (North Atlantic Oscillation) | 20–80°N, 90°W–40°E | EOF 1 | `modes.nao()` |
| **PNA** (Pacific-North American) | 20–85°N, 120°E–120°W | EOF 1 | `modes.pna()` |
| **NPO** (North Pacific Oscillation) | 20–85°N, 120°E–120°W | EOF 2 | `modes.npo()` |
| **SAM** (Southern Annular Mode) | 20–90°S | EOF 1 | `modes.sam()` |
| **PSA1** (Pacific-South American 1) | 20–90°S | EOF 2 | `modes.psa1()` |
| **PSA2** (Pacific-South American 2) | 20–90°S | EOF 3 | `modes.psa2()` |
| **SOI** (Southern Oscillation Index) | Seasonal PSL difference: 30°S–0°N, 70–170°E minus 30°S–0°N, 160–80°W | — | `modes.soi()` |

EOF/PC signs follow CVDP-ncl's per-mode conventions (sign of the pattern at a fixed reference point).

---

## Ocean Circulation

| Metric | Variable | Description | Python Implementation |
|--------|----------|-------------|-----------------------|
| **AMOC** | MOC / msftmz | Leading EOF of the annual Atlantic meridional overturning streamfunction (depth × latitude, area-weighted, south of 33°S excluded): index, pattern, mean, standard deviation, and SST/TAS/ZOS regressions | `ocean_circulation.amoc()` |

---

## Regional Timeseries

Basin- and region-averaged indices computed from monthly fields of SST, PSL, PR, and TAS. See the [NCL methodology page](https://webext.cgd.ucar.edu/Multi-Case/CVDP_repository/cesm2-lens_quadquad_1850-2100/methodology.html) for the full index list.

**Python implementation:** `regional_timeseries.regional_timeseries()`, covering Niño 1+2/3/3.4/4, TNA, TSA, TIO, North Pacific, North Atlantic, Darwin, Tahiti.

The remaining CVDP-ncl SST indices are in `enso.sst_indices()`, and the North/South Pacific Meridional Modes in `modes.npmm()` / `modes.spmm()`. The land-only global PR mean is the global timeseries of land-masked precipitation. The North Pacific Index (PSL, 30–65°N, 160–220°E) is available as the `north_pacific` region. Its NDJFM value uses CVDP-ncl's 5-month running mean sampled at January.

---

## Detrending Options

Applied before most calculations, following CVDP-ncl `remove_trend` (see `trends.detrend()`):

| CVDP-ncl option | Python `detrend=` | Description |
|-----------------|-------------------|-------------|
| None | `"none"` | No trend removed |
| LinearTrend | `"linear"` | Linear trend removed separately for each calendar month; mean kept |
| QuadraticTrend | `"quadratic"` | Quadratic trend removed separately for each calendar month; mean kept |
| 30yrRunningMean | `"highpass30"` | 30-yr weighted running mean (IPCC AR4 weights) subtracted for each calendar month; ends reflected |
| rmEM | — | Ensemble mean removed; applied before the metrics are called, since it needs the other ensemble members |
| rmGMST_EM | — | Ensemble-mean global-mean TAS regressed out; applied before the metrics are called |

---

## Summary Table

| Category | Total Metrics | Implemented | Not Yet Implemented |
|----------|--------------|-------------|---------------------|
| Climatological Mean & Std Dev | 6 | 6 | 0 |
| Trends & Timeseries | 7 | 7 | 0 |
| ENSO | 8 | 8 | 0 |
| Pacific Decadal Modes | 3 | 3 | 0 |
| Atlantic Modes (AMV) | 1 | 1 | 0 |
| Atmospheric Modes | 8 | 8 | 0 |
| Ocean Circulation (AMOC) | 1 | 1 | 0 |
| Regional Timeseries | 1 | 1 | 0 |
| **Total** | **35** | **35** | **0** |

> "Implemented" means computation code exists in `cvdp/metrics/` with unit tests. All modes of variability and ENSO metrics also match real CVDP-ncl output (ERSSTv5, ERA5, GISTEMPv4, GPCP, 1979–2024). Climatology, trends and global means also match CVDP-ncl (trends reported per decade rather than per record length); sea-ice concentration is not yet checked against CVDP-ncl output. Plotting (`cvdp/vis/`) is not yet connected to these functions.

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
