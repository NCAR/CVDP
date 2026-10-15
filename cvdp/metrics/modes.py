"""
cvdp.metrics.modes

Modes of climate variability, grouped by how they are computed:

1. Seasonal PSL EOF modes (NAM, NAO, PNA, NPO, SAM, PSA1, PSA2): thin
   wrappers around :func:`seasonal_eof_mode`.
2. Box-index modes (SOI per season; IPV-Henley, AMV monthly): area-mean
   index, fields regressed onto it.
3. Monthly SST EOF modes (PDO, IPV, NPMM, SPMM): leading PC of SST
   anomalies (for NPMM/SPMM after regressing out ENSO), fields regressed
   onto it.

Follows CVDP-ncl v6.1.0 scripts; definitions: ``docs/llm/metrics.md``.
EOF modes follow NCL's per-mode sign convention: the pattern's sign at a
reference point (e.g. ``nam.ncl:104``), fixed on the raw PC before it is
standardised and fields are regressed onto it.
Bounds are ``(lat_south, lat_north, lon_west, lon_east)``, lon 0–360.

Common parameters
-----------------
psl, ts, pr, tas, zos : xr.DataArray
    Monthly fields, dims (time, lat, lon), cftime ``time``. In sections 2–3,
    ``psl``/``pr``/``tas``/``zos`` are optional; each one supplied adds a
    regression map. CVDP-ncl regresses tas, pr and zos; ``psl`` maps are an
    extra (omit ``psl`` for NCL's output set).
    Regression maps are returned as ``dict[str, DataArray]``, each on its
    field's own grid (as NCL; a Dataset would merge grids onto their union).
    As CVDP-ncl (e.g. ``pdv.ncl:200``), the caller passes ``tas`` masked to
    land (LSMASK != 0) for the section 2–3 modes; ENSO composites take it
    unmasked.
detrend : str
    ``"none"`` or a :data:`cvdp.metrics.trends.DETREND_OPTIONS` value.
seasons : SeasonalDefinition
    Seasons to compute (section 1).
"""
import xarray as xr

from cvdp.metrics.eof import eof, regress
from cvdp.metrics.filters import wgt_runave, lanczos_weights, smth9
from cvdp.metrics.regional_timeseries import box_mean, box_select
from cvdp.metrics.trends import detrended_anomalies as _anomalies
from cvdp.metrics.seasons import SeasonalDefinition, CVDP_SEASONS, NDJFM


# Northern Hemisphere modes add the extended-winter NDJFM season.
NH_SEASONS = CVDP_SEASONS + NDJFM


def _regression_maps(index: xr.DataArray, detrend: str, **fields) -> dict[str, xr.DataArray]:
    """``{var}_regr``: anomalies of each supplied field regressed onto ``index``,
    each on its own grid."""
    return {
        f"{var}_regr": regress(_anomalies(field, detrend), index)
        for var, field in fields.items() if field is not None
    }


def _pc(anom: xr.DataArray, bounds, number: int = 1, sign_point=None, positive: bool = False) -> xr.DataArray:
    """Standardised PC ``number`` over ``bounds``; explained variance in attrs.

    With ``sign_point`` (lat, lon), the raw PC is flipped, before
    standardising, so that ``anom`` regressed onto it is negative at the
    nearest gridpoint (positive if ``positive``). This is the sign NCL gives
    the pattern after regressing (``.ge.0``/``.lt.0`` flip). A missing value
    there leaves the sign arbitrary, as NCL ``ismissing``."""
    pc = eof(box_select(anom, bounds), n=number).sel(mode=number)
    if sign_point is not None:
        lat, lon = sign_point
        slope = float(regress(anom.sel(lat=lat, lon=lon, method="nearest"), pc))
        if (slope < 0) if positive else (slope >= 0):  # NaN: no flip
            pc = -pc
    pc = pc / pc.std("time", ddof=1)  # NCL dim_standardize; mean already 0
    return pc.drop_vars(["mode", "variance_fraction"]).assign_attrs(
        variance_fraction=float(pc["variance_fraction"])
    )


# --- 1. Seasonal PSL EOF modes ----------------------------------------------

def seasonal_eof_mode(
    psl: xr.DataArray,
    name: str,
    bounds: tuple[float, float, float, float],
    eof_number: int,
    seasons: SeasonalDefinition = CVDP_SEASONS,
    detrend: str = "linear",
    sign_point: tuple[float, float] | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    One EOF mode of seasonal-mean anomalies, per season. Shared core of the
    EOF-based modes below, which differ only in domain and EOF number.

    Follows CVDP-ncl (``rmMonAnnCycTLL``, ``remove_trend``,
    ``calculate_eofs``): monthly anomalies are detrended, averaged to one
    value per year with :meth:`~cvdp.metrics.seasons.Season.ncl_annual`
    (equal month weights), the EOFs are computed over ``bounds``, and the
    global field is regressed onto the standardised PC. With ``sign_point``
    the pattern is made negative there (NCL convention); otherwise the sign
    is arbitrary.

    Parameters
    ----------
    psl : xr.DataArray
        Monthly field, dims (time, lat, lon).
    name : str
        Prefix for output variable names.
    bounds : tuple
        EOF domain ``(lat_south, lat_north, lon_west, lon_east)``; see
        :func:`~cvdp.metrics.regional_timeseries.box_select`.
    eof_number : int
        Which EOF to return, numbered from 1.
    seasons : SeasonalDefinition, optional
        Seasons to compute. Defaults to CVDP_SEASONS.
    detrend : str, optional
        ``"none"`` or a :func:`~cvdp.metrics.trends.detrend` method, applied
        to the monthly anomalies. Default ``"linear"``.
    sign_point : tuple, optional
        ``(lat, lon)`` where the pattern is made negative (nearest
        gridpoint). Default None: sign arbitrary.

    Returns
    -------
    patterns : xr.Dataset
        ``{name}_pattern_{season}``, dims (lat, lon), units per std dev of PC.
    timeseries : xr.Dataset
        ``{name}_pc_{season}``, dim (time) of season-years, standardised. The
        ``variance_fraction`` attr holds the fraction of domain variance
        explained.
    """
    anom = _anomalies(psl, detrend)
    patterns, pcs = {}, {}
    for season in seasons:
        seasonal = season.ncl_annual(anom)
        pc = _pc(seasonal, bounds, eof_number, sign_point)
        pcs[f"{name}_pc_{season.name}"] = pc
        patterns[f"{name}_pattern_{season.name}"] = regress(seasonal, pc)
    return xr.Dataset(patterns), xr.Dataset(pcs)


def nam(psl: xr.DataArray, seasons: SeasonalDefinition = NH_SEASONS, detrend: str = "linear"):
    """Northern Annular Mode: EOF 1 over (20, 90, 0, 360). See :func:`seasonal_eof_mode`."""
    return seasonal_eof_mode(psl, "nam", (20, 90, 0, 360), 1, seasons, detrend, (85, 5))


def nao(psl: xr.DataArray, seasons: SeasonalDefinition = NH_SEASONS, detrend: str = "linear"):
    """North Atlantic Oscillation: EOF 1 over (20, 80, 270, 40). See :func:`seasonal_eof_mode`."""
    return seasonal_eof_mode(psl, "nao", (20, 80, 270, 40), 1, seasons, detrend, (70, 350))


def pna(psl: xr.DataArray, seasons: SeasonalDefinition = NH_SEASONS, detrend: str = "linear"):
    """Pacific-North American pattern: EOF 1 over (20, 85, 120, 240). See :func:`seasonal_eof_mode`."""
    return seasonal_eof_mode(psl, "pna", (20, 85, 120, 240), 1, seasons, detrend, (50, 185))


def npo(psl: xr.DataArray, seasons: SeasonalDefinition = NH_SEASONS, detrend: str = "linear"):
    """North Pacific Oscillation: EOF 2 over (20, 85, 120, 240). See :func:`seasonal_eof_mode`."""
    return seasonal_eof_mode(psl, "npo", (20, 85, 120, 240), 2, seasons, detrend, (65, 185))


def sam(psl: xr.DataArray, seasons: SeasonalDefinition = CVDP_SEASONS, detrend: str = "linear"):
    """Southern Annular Mode: EOF 1 over (-90, -20, 0, 360). See :func:`seasonal_eof_mode`."""
    return seasonal_eof_mode(psl, "sam", (-90, -20, 0, 360), 1, seasons, detrend, (-85, 5))


def psa1(psl: xr.DataArray, seasons: SeasonalDefinition = CVDP_SEASONS, detrend: str = "linear"):
    """Pacific-South American pattern 1: EOF 2 over (-90, -20, 0, 360). See :func:`seasonal_eof_mode`."""
    return seasonal_eof_mode(psl, "psa1", (-90, -20, 0, 360), 2, seasons, detrend, (-62, 210))


def psa2(psl: xr.DataArray, seasons: SeasonalDefinition = CVDP_SEASONS, detrend: str = "linear"):
    """Pacific-South American pattern 2: EOF 3 over (-90, -20, 0, 360). See :func:`seasonal_eof_mode`."""
    return seasonal_eof_mode(psl, "psa2", (-90, -20, 0, 360), 3, seasons, detrend, (-60, 280))


# --- 2. Box-index modes ----------------------------------------------------

def soi(
    psl: xr.DataArray,
    seasons: SeasonalDefinition = CVDP_SEASONS,
    detrend: str = "linear",
) -> tuple[xr.Dataset, xr.Dataset]:
    """Southern Oscillation Index, per season (CVDP-ncl ``soi.ncl``): seasonal
    PSL anomaly mean over (-30, 0, 70, 170) minus mean over (-30, 0, 200, 280),
    with seasonal means from :meth:`Season.ncl_annual`.

    Returns ``(patterns, timeseries)``: ``soi_pattern_{season}`` (global
    seasonal PSL regressed onto the index, hPa per hPa) and
    ``soi_index_{season}`` (not standardised, units of ``psl``)."""
    anom = _anomalies(psl, detrend)
    patterns, timeseries = {}, {}
    for season in seasons:
        seasonal = season.ncl_annual(anom)
        index = box_mean(seasonal, (-30, 0, 70, 170)) - box_mean(seasonal, (-30, 0, 200, 280))
        timeseries[f"soi_index_{season.name}"] = index
        patterns[f"soi_pattern_{season.name}"] = regress(seasonal, index)
    return xr.Dataset(patterns), xr.Dataset(timeseries)


def ipv_henley(
    ts: xr.DataArray,
    psl: xr.DataArray = None,
    pr: xr.DataArray = None,
    tas: xr.DataArray = None,
    zos: xr.DataArray = None,
    detrend: str = "linear",
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    """Interdecadal Pacific Variability, Henley et al. (2015) tripole: SST
    anomaly mean over (-10, 10, 170, 270) minus the average of the means over
    (25, 45, 140, 215) and (-50, -15, 150, 200). No EOF.

    Returns ``(index, regr_maps)``: monthly index (units of ``ts``), and
    ``{var}_regr`` per supplied field."""
    anom = _anomalies(ts, detrend)
    north, south = box_mean(anom, (25, 45, 140, 215)), box_mean(anom, (-50, -15, 150, 200))
    index = (box_mean(anom, (-10, 10, 170, 270)) - (north + south) / 2).rename("ipv_henley")
    return index, _regression_maps(index, detrend, ts=ts, psl=psl, pr=pr, tas=tas, zos=zos)


def amv(
    ts: xr.DataArray,
    psl: xr.DataArray = None,
    pr: xr.DataArray = None,
    tas: xr.DataArray = None,
    zos: xr.DataArray = None,
    detrend: str = "linear",
    lowpass_window: int = 121,
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    """Atlantic Multidecadal Variability (CVDP-ncl ``amv.ncl``): SST anomaly
    mean over (0, 60, 280, 360).

    The index's ``low_pass`` coord is its ``lowpass_window``-month running
    mean (NCL ``runave``, missing ends). Maps: ``{var}_regr`` = field
    regressed onto the index (``ts`` map smoothed with 3 passes of
    ``smth9``); ``{var}_regr_lp`` = running-mean field regressed onto
    ``low_pass``.

    Returns ``(index, regr_maps)``: monthly index (units of ``ts``) and maps."""
    anom = _anomalies(ts, detrend)
    index = box_mean(anom, (0, 60, 280, 360)).rename("amv")
    runave = [1.0] * lowpass_window  # equal weights = NCL runave
    low_pass = wgt_runave(index, runave)
    maps = {}
    for var, field in {"ts": ts, "psl": psl, "pr": pr, "tas": tas, "zos": zos}.items():
        if field is None:
            continue
        field_anom = anom if var == "ts" else _anomalies(field, detrend)
        regr = regress(field_anom, index)
        if var == "ts":
            for _ in range(3):
                regr = smth9(regr)
        maps[f"{var}_regr"] = regr
        maps[f"{var}_regr_lp"] = regress(wgt_runave(field_anom, runave), low_pass)
    return index.assign_coords(low_pass=low_pass), maps


# --- 3. Monthly SST EOF modes -----------------------------------------------

def pdo(
    ts: xr.DataArray,
    psl: xr.DataArray = None,
    pr: xr.DataArray = None,
    tas: xr.DataArray = None,
    zos: xr.DataArray = None,
    detrend: str = "linear",
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    """Pacific Decadal Oscillation (CVDP-ncl ``pdv.ncl``): PC 1 of monthly
    SST anomalies over (20, 70, 110, 260). Pattern positive at (57, 220).

    Returns ``(index, regr_maps)``: standardised monthly index
    (``variance_fraction`` in attrs), and ``{var}_regr`` per supplied field
    (``ts_regr`` is the PDO pattern, °C per std dev)."""
    index = _pc(_anomalies(ts, detrend), (20, 70, 110, 260), sign_point=(57, 220), positive=True).rename("pdo")
    return index, _regression_maps(index, detrend, ts=ts, psl=psl, pr=pr, tas=tas, zos=zos)


def ipv_eof(
    ts: xr.DataArray,
    psl: xr.DataArray = None,
    pr: xr.DataArray = None,
    tas: xr.DataArray = None,
    zos: xr.DataArray = None,
    detrend: str = "linear",
    min_years: int = 40,
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    """Interdecadal Pacific Variability, EOF method (CVDP-ncl ``ipv.ncl``):
    PC 1 over (-40, 60, 110, 290) of SST anomalies low-pass filtered with
    217 Lanczos weights, cut-off 1/157 per month (~13 yr), reflected ends.
    Raises ``ValueError`` for records shorter than ``min_years``. Pattern
    negative at (40, 165).

    Returns ``(index, regr_maps)``: standardised monthly index; ``ts_regr``
    from the filtered SST, other ``{var}_regr`` from unfiltered anomalies."""
    if ts.sizes["time"] < 12 * min_years:
        raise ValueError(f"ipv_eof needs at least {min_years} years of data")
    filtered = wgt_runave(_anomalies(ts, detrend), lanczos_weights(217, 1 / 157), kopt=1)
    index = _pc(filtered, (-40, 60, 110, 290), sign_point=(40, 165)).rename("ipv")
    maps = _regression_maps(index, detrend, psl=psl, pr=pr, tas=tas, zos=zos)
    return index, maps | {"ts_regr": regress(filtered, index)}


def _remove_enso(anom: xr.DataArray) -> xr.DataArray:
    """SST anomalies minus their regression onto ENSO, the standardised PC 1
    over (-20, 20, 220, 290), as CVDP-ncl ``sst.indices.ncl`` before
    NPMM/SPMM. Pattern times PC does not depend on the PC's sign."""
    pc = _pc(anom, (-20, 20, 220, 290))
    return anom - regress(anom, pc) * pc


def npmm(ts: xr.DataArray, detrend: str = "linear") -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    """North Pacific Meridional Mode (CVDP-ncl ``sst.indices.ncl``): PC 1
    over (-20, 30, 175, 265) of SST anomalies with ENSO regressed out.
    Pattern positive at (22, 240).

    Returns ``(index, regr_maps)``: standardised monthly index; ``ts_regr`` is
    the ENSO-removed SST regressed onto it."""
    resid = _remove_enso(_anomalies(ts, detrend))
    index = _pc(resid, (-20, 30, 175, 265), sign_point=(22, 240), positive=True).rename("npmm")
    return index, {"ts_regr": regress(resid, index)}


def spmm(ts: xr.DataArray, detrend: str = "linear") -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    """South Pacific Meridional Mode (CVDP-ncl ``sst.indices.ncl``): PC 1
    over (-35, -20, 180, 290) of SST anomalies with ENSO regressed out.
    Pattern positive at (-25, 280).

    Returns ``(index, regr_maps)`` as :func:`npmm`."""
    resid = _remove_enso(_anomalies(ts, detrend))
    index = _pc(resid, (-35, -20, 180, 290), sign_point=(-25, 280), positive=True).rename("spmm")
    return index, {"ts_regr": regress(resid, index)}
