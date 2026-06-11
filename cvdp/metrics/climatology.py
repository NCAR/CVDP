"""
cvdp.metrics.climatology

Seasonal climatological means and standard deviations, computed without
removing the annual cycle. Seasons are supplied as a SeasonalDefinition,
so overlapping seasons (DJF vs. JFM, ANN) are supported.
"""
import xarray as xr

from cvdp.metrics.seasons import SeasonalDefinition, CVDP_SEASONS


def get_seasonal_statistics(
    ds: xr.Dataset,
    seasons: SeasonalDefinition = CVDP_SEASONS,
) -> xr.Dataset:
    """
    Day-weighted seasonal mean and standard deviation for every variable.

    Parameters
    ----------
    ds : xr.Dataset
        Monthly fields with a cftime ``time`` dimension.
    seasons : SeasonalDefinition, optional
        Seasons to compute. Defaults to CVDP_SEASONS.

    Returns
    -------
    xr.Dataset
        ``{var}_mean`` and ``{var}_std`` per input variable, stacked on a
        ``season`` dimension.
    """
    mean = seasons.mean(ds)
    std = seasons.std(ds)
    return xr.Dataset(
        {f"{var}_mean": mean[var] for var in ds.data_vars}
        | {f"{var}_std": std[var] for var in ds.data_vars}
    )
