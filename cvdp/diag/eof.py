#!/usr/bin/env python3
"""
climatology.py

CVDP functions for calculating climatological means and standard deviations.
License: MIT
"""
import xarray


CLIMATOLOGY_SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "JFM": [1, 2, 3],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "JAS": [7, 8, 9],
    "SON": [9, 10, 11],
    "ANN": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
}


from geocat.comp import eofunc_eofs, eofunc_pcs, month_to_season
import numpy as np

def get_eof(ds, season, latlon_dict, neof):
    """
    Calculate the Empirical Orthogonal Functions (EOF), Principal Components (PCS)

    Parameters
    ----------
    ds : list of float
        A list of numerical values.
    season : str
        Abbreviation of season, ie DJF, SON, etc.
    latlon_dict : dictionary
        A dict of lat/lon coordinates
    neof : int
        number of EOFs

    Returns
    -------
    eofs : xarray.DataArray
        Seasonal avergae of sea level pressure
    pcs : xarray.DataArray
        Time series of principal components
    SLP : xarray.DataArray
        Seasonal avergae of sea level pressure

    Raises
    ------
    ValueError
        If `values` is empty.
    """

    latS = latlon_dict['s']
    latN = latlon_dict['n']
    lonE = latlon_dict['e']
    lonW = latlon_dict['w']

    #neof = 3  # number of EOFs

    # To facilitate data subsetting
    ds2 = ds.copy()

    #season = "DJF"
    SLP = month_to_season(ds2, season)

    clat = SLP['lat'].astype(np.float64)
    clat = np.sqrt(np.cos(np.deg2rad(clat)))

    wSLP = SLP
    wSLP = SLP * clat

    # For now, metadata for slp must be copied over explicitly; it is not preserved by binary operators like multiplication.
    wSLP.attrs = ds2.attrs

    xw = wSLP.sel(lat=slice(latS, latN))

    # Transpose data to have 'time' in the first dimension
    # as `eofunc` functions expects so for xarray inputs for now
    xw_slp = xw.transpose('time', 'lat', 'lon')

    # Doesn't look like we use the actual eofs functions...
    eofs = eofunc_eofs(xw_slp, neofs=neof, meta=True)

    # Gather time series calcuation
    pcs = eofunc_pcs(xw_slp, npcs=neof, meta=True)

    # Change the sign of the second EOF and its time-series for
    # consistent visualization purposes. See this explanation:
    # https://www.ncl.ucar.edu/Support/talk_archives/2009/2015.html
    # about that EOF signs are arbitrary and do not change the physical
    # interpretation.
    for i in range(neof):
        if i == 1:
            pcs[i, :] *= (-1)
            eofs[i, :, :] *= (-1)

    # Sum spatial weights over the area used.
    nLon = xw.sizes["lon"]

    # Bump the upper value of the slice, so that latitude values equal to latN are included.
    clat_subset = clat.sel(lat=slice(latS, latN + 0.01))
    weightTotal = clat_subset.sum() * nLon
    pcs = pcs / weightTotal

    return eofs, pcs, SLP