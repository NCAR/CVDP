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

from cvdp.metrics.regional_timeseries import box_mean, monthly_anomalies, REGIONS


NINO34_BOUNDS = (-5.0, 5.0, 190.0, 240.0)  # 5°S–5°N, 170–120°W


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
    pass


def nino34_power_spectrum(
    nino34: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Power spectrum of the Niño3.4 index.

    Parameters
    ----------
    nino34 : xr.DataArray
        Monthly Niño3.4 index, dim (time). As returned by
        :func:`nino34_index`. The time coordinate must use cftime objects.

    Returns
    -------
    power : xr.DataArray
        Power spectral density, dim (frequency). Units: variance per cycle/month.
    frequency : xr.DataArray
        Frequencies corresponding to ``power``, dim (frequency).
        Units: cycles per month.
    """
    pass


def nino34_wavelet(
    nino34: xr.DataArray,
    wavenumber: int = 6,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Morlet wavelet transform of the Niño3.4 index.

    Significance is assessed against a chi-square test at the 95% level.
    The cone of influence (COI) marks regions affected by edge effects.

    Parameters
    ----------
    nino34 : xr.DataArray
        Monthly Niño3.4 index, dim (time). As returned by
        :func:`nino34_index`. The time coordinate must use cftime objects.
    wavenumber : int, optional
        Morlet wavenumber parameter controlling frequency/time resolution
        trade-off. Default 6 (matches NCL CVDP).

    Returns
    -------
    power : xr.DataArray
        Wavelet power spectrum, dims (period, time). Units: variance.
    significance : xr.DataArray
        Boolean mask of grid points significant at 95%, dims (period, time).
    coi : xr.DataArray
        Cone of influence, dim (time). Periods above the COI are affected
        by edge effects. Units: months.
    """
    pass


def enso_composites(
    ts: xr.DataArray,
    psl: xr.DataArray,
    pr: xr.DataArray,
    tas: xr.DataArray,
    nino34: xr.DataArray,
    threshold: float = 1.0,
) -> xr.Dataset:
    """
    Spatial composites of SST, PSL, PR, and TAS during El Niño and La Niña events.

    Events are defined by the December Niño3.4 value (smoothed) exceeding
    ``+threshold`` std dev (El Niño) or ``-threshold`` std dev (La Niña).

    Parameters
    ----------
    ts : xr.DataArray
        Monthly sea surface temperature, dims (time, lat, lon).
        The time coordinate must use cftime objects.
    psl : xr.DataArray
        Monthly sea level pressure, dims (time, lat, lon).
        The time coordinate must use cftime objects.
    pr : xr.DataArray
        Monthly precipitation rate, dims (time, lat, lon).
        The time coordinate must use cftime objects.
    tas : xr.DataArray
        Monthly 2-m air temperature, dims (time, lat, lon).
        The time coordinate must use cftime objects.
    nino34 : xr.DataArray
        Monthly Niño3.4 index, dim (time). As returned by
        :func:`nino34_index`. The time coordinate must use cftime objects.
    threshold : float, optional
        Number of standard deviations used to define events. Default 1.0.

    Returns
    -------
    xr.Dataset
        Variables ``{var}_elnino`` and ``{var}_lanina`` for var in
        ``["ts", "psl", "pr", "tas"]``, each with dims (lat, lon).
        Dataset attributes include event counts for El Niño and La Niña.
    """
    pass


def enso_hovmoller(
    ts: xr.DataArray,
    nino34: xr.DataArray,
    threshold: float = 1.0,
    lat_bounds: tuple[float, float] = (-3.0, 3.0),
) -> xr.Dataset:
    """
    El Niño and La Niña Hovmöller diagrams of equatorial SST anomalies.

    Meridional average is taken over ``lat_bounds``. Events are defined
    identically to :func:`enso_composites`. Composites span Jan of year 0
    to May of year +2 (18 months).

    Parameters
    ----------
    ts : xr.DataArray
        Monthly sea surface temperature, dims (time, lat, lon).
        The time coordinate must use cftime objects.
    nino34 : xr.DataArray
        Monthly Niño3.4 index, dim (time). As returned by
        :func:`nino34_index`. The time coordinate must use cftime objects.
    threshold : float, optional
        Standard deviation threshold for event definition. Default 1.0.
    lat_bounds : tuple[float, float], optional
        (south, north) latitude limits for the meridional average.
        Default ``(-3.0, 3.0)``.

    Returns
    -------
    xr.Dataset
        Variables ``ts_elnino_hovmoller`` and ``ts_lanina_hovmoller``,
        each with dims (lead_month, lon) where ``lead_month`` ranges from
        0 (Jan yr0) to 16 (May yr+2).
    """
    pass


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
    pass
