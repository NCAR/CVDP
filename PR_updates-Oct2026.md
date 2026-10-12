# PRECT rename, ANN season, EOF/NPI ensemble fixes, and difference-plot fixes

**From:** `richling-ucar:updates-Oct2026` → **into:** `NCAR:updates-jr-Oct2026`

Follow-up to #37. It adds the 14 commits made after #37 was merged (from `d2a2997` to `2b029bb`). The branch has no conflicts with `NCAR:main`, which has no new commits since `c731979`.

## Summary

- Rename the CVDP precipitation variable from `pr` to `prect`, including variable defaults, image names and webpage links.
- Add the annual (`ANN`) season.
- Fix the ensemble-mean and sign handling for EOF and NPI patterns.
- Fix the difference panels in the summary plots.
- Make the reference-to-simulation regridding more robust.
- Fix a crash in the NCL colormap loader.
- Add one example config per variable, plus a temporary rule that picks `vns` from the config.

## Changes

### Variables and config
- **`pr` → `prect`:** `variable_defaults.yaml` now uses `prect`, with the `precip3_16lev` colormap and `MPL_RdYlGn` for differences. All four HTML templates now link to `prect_*`, `*.prect.*` and `*prectreg*` images, and their labels read `PRECT`.
- **`ts` → `sst`** in `variable_defaults.yaml`. The SST `spatialstddev` and `trends` levels are commented out for now, so the plots use default levels.
- **SST clamp** in `file_creation.py`: SST values below -1.8 °C are set to -1.8 °C, as in NCL CVDP. This removes sea-ice and land skin temperatures that come in through `TS`.
- **Kelvin units:** `deg_k` and `degK` are now also recognized as Kelvin.
- **New example configs:** `example_config_PRECT.yaml`, `example_config_PSL.yaml`, `example_config_SST.yaml` and `example_config_TAS.yaml`.
- **Temporary rule in `cli.py`:** `vns` is set to one variable, taken from the first case under `Data` in the config and mapped through `VARNAME_MAP`. This should be removed once one run can handle several variables.

### Annual season
- `ANN` added to `utils.season_list`.
- `ANN` in `diag/eof.py` and `diag/linear_trends.py` was missing month 11. It now has all 12 months.
- `calculate_eof` uses a days-in-month weighted annual mean for `ANN`, because `month_to_season` has no annual option.
- `seasonal_timeseries`: the `ANN` field is now the standard deviation of the annual-mean anomalies across years (`ddof=0`). Before, it was the annual mean timeseries.

### EOF / NPI patterns (`analysis.py`, `utils.py`)
- **Fixed EOF sign:** EOF signs are arbitrary, so the sign is now forced at fixed points to match NCL CVDP: NAM (85N, 5E), SAM (85S, 5E), PSA1 (62S, 210E), PSA2 (60S, 280E). Before, datasets with opposite signs gave differences that doubled the pattern instead of showing the bias.
- **Ensemble-mean pattern:** for both EOF and NPI, this is now the mean of the individual member patterns. Before, it was the pattern of the ensemble-mean field. The EOF PC of the ensemble mean is still kept for the timeseries plots.

### Regridding (`analysis.interp_diff`)
- Rewritten. Regridding now happens if either lat or lon differs. Before, it only happened when both differed, and xarray's label alignment could silently drop points that didn't match.
- Grids that match to within floating-point noise reuse the simulation's coordinates, with no regridding. Grids that match exactly return `None`.

### Plotting
- **Summary difference panels** in `global_plots.py` and `polar_plots.py` are now computed for each row from that row's sim and ref. The panel always matches columns 0 and 1, and its default levels come from the difference itself.
- **Individual-member difference plots** now get per-member differences (`sims × refs`) instead of ensemble-mean differences.
- **Symmetric default levels:** when no difference levels are configured, the defaults are symmetric about zero (21 levels), so zero sits at the center of the diverging colormap.
- **Tick bug fix in `get_ticks`:** the check was `if ticks is not None`, so configured ticks were always replaced by the levels. It is now `if ticks is None`. **Configured `*_ticks_*` values now take effect, so some colorbars will look different.**
- **Colormap loader fix in `colormaps.py`:** when `NCARG_ROOT` was set but the colormap wasn't in the NCL folder, the loader crashed with `UnboundLocalError`. It now tries the NCL URL next and, if the colormap isn't found there either, raises a clear `FileNotFoundError` that names it.
- `np.nanmin` and `np.nanmax` are used for the default level range in the individual-member difference plots.
- Removed unused imports from `global_plots.py` and `vis_utils.py`, and added the imports that `vis_utils.py` was missing (`ccrs`, `cfeature`, `af`).

## Testing
- Ran `cli.py` on Derecho in the `cvdp-dev` env with `example_config_PRECT.yaml`. The summary, individual-member and individual-member difference plots were produced. Rebuilding the webpages from that output linked all 54 `prect_*.png` files from `index.html` and `index_indmem.html`.
- Checked that the colormap loader reads local and `$NCARG_ROOT` colormaps and raises a clear error for an unknown name.
- <!-- add the PSL / SST / TAS runs and NCL-comparison checks here -->

## Known follow-ups (not in this PR)
- Remove the temporary single-variable rule for `vns` and the stacked `vns = [...]` lines in `cli.py`.
- `VARNAME_MAP` is defined in three places (`cli.py`, `file_io.py`, `file_creation.py`). Keep one copy.
- Debug prints to remove: `sim_datasets` in `cli.py`, `sim_unit` and `arr_min`/`arr_max` in `global_plots.py`, and `PATH_ROOT_DIR` in `definitions.py`.
- The `valid_time` fallback in `file_creation.py` (from #37) uses bare `except:` and `sys.exit(0)`. A failed read therefore exits with a success code; it should raise instead.
- The `arr_diffs` / `arr_diff` arguments of `global_ensemble_plot` and `polar_ensemble_plot` are no longer used and can be removed.
- Finalize the SST `spatialstddev` and `trends` levels in `variable_defaults.yaml`.
- The ENSO-composite and AMV/PDV regression links assume future file names `nino34.spatialcomp.prect.*` and `*.prectreg.*`. Those plots aren't produced in Python yet.
- `cvdp/ncl_compare_indmem_index.html` is a generated file but is tracked in git.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
