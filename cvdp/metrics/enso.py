"""
cvdp.metrics.enso

ENSO diagnostics derived from the Niño3.4 SST index
(area-averaged SST anomalies over 5°S–5°N, 120–170°W).

El Niño / La Niña events are defined by the December Niño3.4 value
(3-point binomial smoothed) exceeding ±1 standard deviation.
"""
import cftime
import numpy as np
import xarray as xr
from scipy.stats import chi2

from cvdp.metrics.filters import wgt_runave
from cvdp.metrics.regional_timeseries import box_mean, box_select, detrended_anomalies, monthly_anomalies, REGIONS


NINO34_BOUNDS = (-5.0, 5.0, 190.0, 240.0)  # 5°S–5°N, 170–120°W
SST_INDEX_REGIONS = ["nino12", "nino3", "nino34", "nino4", "tna", "tsa", "tio"]


def nino34_index(
    ts: xr.DataArray,
    time_start: cftime.datetime = None,
    time_end: cftime.datetime = None,
    smooth: bool = True,
) -> xr.DataArray:
    """
    Compute the Niño3.4 SST anomaly index.

    Area-averaged SST anomalies over 5°S–5°N, 120–170°W. Anomalies are
    computed relative to the full-period monthly climatology within
    [time_start, time_end].

    Parameters
    ----------
    ts : xr.DataArray
        Monthly sea surface temperature, dims (time, lat, lon).
        The time coordinate must use cftime objects.
    time_start : cftime.datetime, optional
        Start of the analysis period. Defaults to the first time step of ``ts``.
    time_end : cftime.datetime, optional
        End of the analysis period. Defaults to the last time step of ``ts``.
    smooth : bool, optional
        Apply a 3-point binomial filter (weights [0.25, 0.5, 0.25]) to the
        index before returning. Default ``True``.

    Returns
    -------
    xr.DataArray
        Monthly Niño3.4 index, dim (time). Units: °C (or same as ``ts``).
    """
    ts = ts.sel(time=slice(time_start, time_end))
    index = monthly_anomalies(box_mean(ts, NINO34_BOUNDS))
    if smooth:
        # 3-point binomial filter [0.25, 0.5, 0.25]; endpoints keep the
        # unsmoothed value so no time steps are lost.
        smoothed = 0.25 * index.shift(time=1) + 0.5 * index + 0.25 * index.shift(time=-1)
        index = smoothed.fillna(index)
    index.name = "nino34"
    return index


def nino34_monthly_stddev(
    nino34: xr.DataArray,
) -> xr.DataArray:
    """
    Standard deviation of the Niño3.4 index by calendar month.

    Parameters
    ----------
    nino34 : xr.DataArray
        Monthly Niño3.4 index, dim (time). As returned by
        :func:`nino34_index`. The time coordinate must use cftime objects.

    Returns
    -------
    xr.DataArray
        Standard deviation for each calendar month, dim (month) with values
        1–12. Units same as ``nino34``.
    """
    std = nino34.groupby("time.month").std("time")
    std.name = "nino34_monthly_stddev"
    return std


def nino34_autocorrelation(
    nino34: xr.DataArray,
    max_lag: int = 24,
) -> xr.DataArray:
    """
    Lag-autocorrelation of the Niño3.4 index.

    Parameters
    ----------
    nino34 : xr.DataArray
        Monthly Niño3.4 index, dim (time). As returned by
        :func:`nino34_index`. The time coordinate must use cftime objects.
    max_lag : int, optional
        Maximum lag in months. Default 24.

    Returns
    -------
    xr.DataArray
        Autocorrelation coefficients, dim (lag) ranging from ``-max_lag`` to
        ``+max_lag`` in steps of 1 month. Dimensionless.
    """
    x = nino34 - nino34.mean("time")
    denom = (x * x).sum("time")
    lags = list(range(0, max_lag + 1))
    pos = [(x * x.shift(time=-k)).sum("time") / denom for k in lags]
    pos = xr.concat(pos, dim="lag").assign_coords(lag=lags)
    # Autocorrelation of a real series is symmetric: r(-k) == r(k).
    neg = pos.sel(lag=slice(1, None)).assign_coords(lag=[-k for k in lags[1:]])
    acf = xr.concat([neg.sortby("lag"), pos], dim="lag")
    acf.name = "nino34_autocorrelation"
    return acf


def nino34_power_spectrum(nino34: xr.DataArray) -> xr.DataArray:
    """Power spectrum with Markov red-noise confidence curves, as CVDP-ncl
    ``specx_anal(x, 0, (7*nyr)/100, 0.1)`` + ``specx_ci(sdof, 0.95, 0.99)``.

    Mean removed, 10% split-cosine-bell taper, periodogram smoothed with a
    modified Daniell filter (none for records under 43 yr), scaled so its
    trapezoid integral equals the variance.

    Returns DataArray dims (curve, frequency): curves ``spectrum``,
    ``red_noise``, ``red_noise_95``, ``red_noise_99``; frequency in cycles
    per month (f=0 excluded); attrs ``dof``, ``bandwidth``, ``xlag1``."""
    x = nino34.values.astype(float)
    n = x.size
    xvaro = (np.sum(x**2) - np.sum(x) ** 2 / n) / n
    x = x - x.mean()
    # Split-cosine-bell taper (NCL specx_dp.f; PI truncated as in the Fortran).
    pct = float(np.float32(0.1))
    m = max(1, int(pct * n + 0.5) // 2)
    w = 0.5 - 0.5 * np.cos(3.141592653589 / m * (np.arange(1, m + 1) - 0.5))
    x[:m] *= w
    x[n - m:] *= w[::-1]
    cftapr = 0.5 * (128 - 93 * pct) / (8 - 5 * pct) ** 2
    # Lag-1 autocorrelation of the tapered series (NCL statx_dp.f).
    xm = x.mean()
    var = max((np.sum(x**2) - np.sum(x) ** 2 / n) / (n - 1), 0.0)
    xlag1 = np.sum((x[1:] - xm) * (x[:-1] - xm)) / (n - 2) / var
    # Periodogram (FFTPACK scaling; the even-N Nyquist bin is a quarter).
    power = np.abs(2 * np.fft.rfft(x)[1:n // 2 + 1] / n) ** 2
    if n % 2 == 0:
        power[-1] /= 4
    jave = 7 * (n // 12) // 100
    wgtsq = 1.0
    if abs(jave) > 2:
        nj = 2 * (abs(jave) // 2) + 1
        dw = np.r_[0.5, np.ones(nj - 2), 0.5] / (nj - 1)
        wgtsq = np.sum(dw**2)
        power = np.correlate(np.pad(power, nj // 2, mode="reflect"), dw, mode="valid")
    spec = power * xvaro / ((0.5 * power[0] + power[1:-1].sum() + 0.5 * power[-1]) / n)
    dof = 2 / (cftapr * wgtsq)
    frequency = np.arange(1, n // 2 + 1) / n
    # specx_ci (NCL shea_util.ncl): twopi is the NCL float 2.*3.14159.
    twopi = float(np.float32(2.0 * 3.14159))
    markov = 1 / (1 + xlag1**2 - 2 * xlag1 * np.cos(twopi * frequency))
    red = markov * spec.sum() / markov.sum()
    curves = [spec, red, red * chi2.ppf(0.95, dof) / dof, red * chi2.ppf(0.99, dof) / dof]
    return xr.DataArray(
        np.array(curves),
        coords={"curve": ["spectrum", "red_noise", "red_noise_95", "red_noise_99"], "frequency": frequency},
        dims=("curve", "frequency"),
        name="nino34_power_spectrum",
        attrs={"dof": dof, "bandwidth": dof / n * 0.5, "xlag1": xlag1},
    )


def nino34_wavelet(
    nino34: xr.DataArray,
    wavenumber: float = 6.0,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Morlet wavelet transform, as CVDP-ncl ``wavelet(x, 0, 1/12, 6, dt,
    1/12, jtot, N, 1, 0, 0.05, 0)`` (Torrence & Compo 1998): dt = 1/12 yr,
    12 scales per octave, no padding, red-noise (lag-1) 95% significance.

    Returns ``(power, significance, coi)``: ``power`` = |W|^2 (units of
    ``nino34`` squared) and ``significance`` = power / 95% red-noise level
    (> 1 is significant), dims (period, time), period in years; ``coi``
    (cone of influence, years), dim (time)."""
    y = nino34.values.astype(float)
    n = y.size
    dt = s0 = dj = float(np.float32(1 / 12))
    jtot = 1 + int(np.float32(np.log10(np.float32(n * np.float32(dt) / np.float32(s0))) / np.float32(dj))
                   / np.float32(np.log10(2)))
    ymean = y.mean()
    yvar = max((np.sum(y**2) - np.sum(y) ** 2 / n) / n, 0.0)
    lag1 = np.sum((y[1:] - ymean) * (y[:-1] - ymean)) / (n - 1) / yvar
    yf = np.fft.fft(y - ymean) / n
    steps = np.arange(n)
    steps[steps > n // 2] -= n
    k = 2 * np.pi * steps / (n * dt)
    scale = s0 * 2 ** (np.arange(jtot) * dj)
    daughter = (np.sqrt(2 * np.pi * scale[:, None] / dt) * np.pi**-0.25
                * np.exp(-0.5 * (scale[:, None] * k - wavenumber) ** 2) * (k >= 0))
    power = np.abs(n * np.fft.ifft(daughter * yf, axis=1)) ** 2
    fourier_factor = 4 * np.pi / (wavenumber + np.sqrt(2 + wavenumber**2))
    period = scale * fourier_factor
    coi = fourier_factor / np.sqrt(2) * dt * np.minimum(steps % n, n - 1 - np.arange(n))
    r = max(lag1, 0.0)
    red = np.mean((y - ymean) ** 2) * (1 - r**2) / (1 - 2 * r * np.cos(2 * np.pi * dt / period) + r**2)
    signif = red * chi2.ppf(0.95, 2) / 2
    coords = {"period": period, "time": nino34["time"]}
    return (
        xr.DataArray(power, coords=coords, dims=("period", "time"), name="nino34_wavelet_power"),
        xr.DataArray(power / signif[:, None], coords=coords, dims=("period", "time"), name="nino34_wavelet_significance"),
        xr.DataArray(coi, coords={"time": nino34["time"]}, dims="time", name="nino34_wavelet_coi"),
    )


# Composite seasons: month offset from January of the event year (year 0) of
# the centre month of a 3-month running mean, as CVDP-ncl sst.indices.ncl.
COMPOSITE_SEASONS = {"JJA0": 6, "SON0": 9, "DJF1": 12, "MAM1": 15}


def enso_events(
    ts: xr.DataArray,
    detrend: str = "linear",
    threshold: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """El Niño and La Niña years, as CVDP-ncl ``sst.indices.ncl``.

    Niño3.4 from SST anomalies (then ``detrend``), smoothed with [1,2,1]
    (NCL ``wgt_runave``, reflected ends); December values of every year but
    the last, standardised. Events: ``>= threshold`` (El Niño) or
    ``<= -threshold`` (La Niña). Records must start in January.

    Returns ``(elnino_years, lanina_years)``: calendar years of the December."""
    nino34 = box_mean(detrended_anomalies(ts, detrend), NINO34_BOUNDS)
    smoothed = wgt_runave(nino34, [1.0, 2.0, 1.0], kopt=1)
    dec = smoothed.sel(time=smoothed["time"].dt.month == 12).isel(time=slice(None, -1))
    dec = (dec - dec.mean()) / dec.std(ddof=1)
    years = dec["time"].dt.year.values
    return years[(dec >= threshold).values], years[(dec <= -threshold).values]


def _by_month(da: xr.DataArray) -> xr.DataArray:
    """Relabel ``time`` as months since January of year 0 (year * 12 + month - 1)."""
    return da.assign_coords(time=(da["time"].dt.year * 12 + da["time"].dt.month - 1).values)


def enso_composites(
    ts: xr.DataArray,
    psl: xr.DataArray = None,
    pr: xr.DataArray = None,
    tas: xr.DataArray = None,
    detrend: str = "linear",
    threshold: float = 1.0,
) -> xr.Dataset:
    """Seasonal El Niño / La Niña composites, as CVDP-ncl ``sst.indices.ncl``.

    Each supplied field's anomalies (then ``detrend``) get a 3-month running
    mean, averaged over the :func:`enso_events` years at JJA0, SON0, DJF1,
    MAM1. Fewer than 2 events gives NaN.

    Returns Dataset with ``{var}_elnino``, ``{var}_lanina``,
    ``{var}_elnino_minus_lanina``, dims (season, lat, lon); attrs
    ``elnino_events``, ``lanina_events``."""
    elnino, lanina = enso_events(ts, detrend, threshold)

    def composite(field, years):
        # NCL replaces the first running-mean step; no composite season uses it.
        run3 = _by_month(wgt_runave(detrended_anomalies(field, detrend), [1.0, 1.0, 1.0]))
        comp = xr.concat(
            [run3.sel(time=years * 12 + offset).mean("time") for offset in COMPOSITE_SEASONS.values()],
            dim="season",
        ).assign_coords(season=list(COMPOSITE_SEASONS))
        return comp if len(years) >= 2 else comp * np.nan

    out = {}
    for var, field in {"ts": ts, "psl": psl, "pr": pr, "tas": tas}.items():
        if field is None:
            continue
        out[f"{var}_elnino"] = composite(field, elnino)
        out[f"{var}_lanina"] = composite(field, lanina)
        out[f"{var}_elnino_minus_lanina"] = out[f"{var}_elnino"] - out[f"{var}_lanina"]
    return xr.Dataset(out, attrs={"elnino_events": len(elnino), "lanina_events": len(lanina)})


def enso_hovmoller(
    ts: xr.DataArray,
    detrend: str = "linear",
    threshold: float = 1.0,
    lat_bounds: tuple[float, float] = (-3.0, 3.0),
) -> xr.Dataset:
    """El Niño / La Niña Hovmöllers of equatorial SST anomalies, as CVDP-ncl
    ``sst.indices.ncl``: unweighted mean over ``lat_bounds``, 120–280°E,
    averaged over :func:`enso_events` years from Jan of year 0 to May of
    year +2. Events in the first two or last three years are skipped.
    Raises ``ValueError`` for records shorter than 15 years.

    Returns Dataset with ``ts_elnino_hovmoller``, ``ts_lanina_hovmoller``,
    dims (lead_month, lon), lead_month 0–28; attrs ``elnino_events``,
    ``lanina_events`` (events used)."""
    years = ts["time"].dt.year
    first, last = int(years.min()), int(years.max())
    if last - first + 1 < 15:
        raise ValueError("enso_hovmoller needs at least 15 years of data")
    anom = detrended_anomalies(ts, detrend)
    band = _by_month(box_select(anom, (lat_bounds[0], lat_bounds[1], 120, 280)).mean("lat"))

    def hovmoller(event_years):
        event_years = [y for y in event_years if first + 2 <= y <= last - 3]
        windows = [band.sel(time=np.arange(y * 12, y * 12 + 29)).assign_coords(time=np.arange(29))
                   for y in event_years]
        hov = xr.concat(windows, dim="event").mean("event") if windows else band.isel(time=slice(0, 29)) * np.nan
        return hov.rename(time="lead_month").assign_coords(lead_month=np.arange(29)), len(event_years)

    elnino, lanina = enso_events(ts, detrend, threshold)
    hov_en, n_en = hovmoller(elnino)
    hov_ln, n_ln = hovmoller(lanina)
    return xr.Dataset(
        {"ts_elnino_hovmoller": hov_en, "ts_lanina_hovmoller": hov_ln},
        attrs={"elnino_events": n_en, "lanina_events": n_ln},
    )


def sst_indices(
    ts: xr.DataArray,
    time_start: cftime.datetime = None,
    time_end: cftime.datetime = None,
) -> xr.Dataset:
    """
    Regional SST anomaly indices.

    Computes area-weighted SST anomalies for standard ENSO and climate
    index boxes. Anomalies are relative to the full-period monthly climatology
    within [time_start, time_end].

    Regions computed
    ----------------
    - Niño 1+2 : 0–10°S, 90–80°W
    - Niño 3   : 5°N–5°S, 150–90°W
    - Niño 3.4 : 5°N–5°S, 170–120°W
    - Niño 4   : 5°N–5°S, 160°E–150°W
    - Tropical North Atlantic (TNA) : 5–25°N, 55–15°W
    - Tropical South Atlantic (TSA) : 0–20°S, 30°W–10°E
    - Tropical Indian Ocean (TIO) : 20°S–20°N, 40–110°E

    Parameters
    ----------
    ts : xr.DataArray
        Monthly sea surface temperature, dims (time, lat, lon).
        The time coordinate must use cftime objects.
    time_start : cftime.datetime, optional
        Start of the analysis period. Defaults to the first time step of ``ts``.
    time_end : cftime.datetime, optional
        End of the analysis period. Defaults to the last time step of ``ts``.

    Returns
    -------
    xr.Dataset
        Variables ``nino12``, ``nino3``, ``nino34``, ``nino4``, ``tna``,
        ``tsa``, ``tio``, each with dim (time). Units same as ``ts``.
    """
    ts = ts.sel(time=slice(time_start, time_end))
    out = {
        region: monthly_anomalies(box_mean(ts, REGIONS[region]))
        for region in SST_INDEX_REGIONS
    }
    return xr.Dataset(out)
