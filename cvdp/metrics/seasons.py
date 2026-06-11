"""
cvdp.seasons

Season definitions shared by all CVDP metrics. A ``Season`` names the calendar
months a statistic is computed over; a ``SeasonalDefinition`` is the ordered
set of Seasons a pipeline computes. Both plug directly into xarray through
boolean selection (``sel``) and per-year grouping (``groupby``), so
overlapping seasons (DJF vs. JFM, ANN) and cross-year seasons (DJF, NDJFM)
are all supported -- which a month->label dict cannot express.

Pipelines define their seasons once, upstream::

    from cvdp.seasons import CVDP_SEASONS, NDJFM

    get_seasonal_statistics(ds, seasons=CVDP_SEASONS)
    trend_maps(psl, seasons=CVDP_SEASONS + NDJFM)
"""
from dataclasses import dataclass
import xarray as xr


@dataclass(frozen=True)
class Season:
    """A named set of calendar months, ordered from the start of the season.
    May cross the year boundary, e.g. ``Season("DJF", (12, 1, 2))``."""
    name: str
    months: tuple[int, ...]

    @property
    def crosses_year(self) -> bool:
        return self.months[0] > self.months[-1]

    def mask(self, time: xr.DataArray) -> xr.DataArray:
        """Boolean mask of times falling within this season."""
        return time.dt.month.isin(list(self.months))

    def sel(self, obj):
        """Subset a DataArray/Dataset to this season's months."""
        return obj.sel(time=self.mask(obj["time"]))

    def years(self, time: xr.DataArray) -> xr.DataArray:
        """Season-year labels: for cross-year seasons, all months are assigned
        the year of the final month (e.g., DJF 1990 = Dec 1989 + Jan/Feb 1990)."""
        year = time.dt.year
        if self.crosses_year:
            late = [m for m in self.months if m >= self.months[0]]
            year = year + time.dt.month.isin(late)
        return year.rename("season_year")

    def groupby(self, obj):
        """Group this season's months by season-year; chain any xarray
        reduction, e.g. ``season.groupby(da).mean()``."""
        sel = self.sel(obj)
        return sel.groupby(self.years(sel["time"]))

    def mean(self, obj):
        """Day-weighted climatological mean over all years."""
        sel = self.sel(obj)
        return sel.weighted(sel["time"].dt.days_in_month).mean("time")

    def std(self, obj):
        """Day-weighted climatological standard deviation over all years."""
        sel = self.sel(obj)
        return sel.weighted(sel["time"].dt.days_in_month).std("time")

    def annual(self, obj):
        """Day-weighted seasonal mean for each year, on an integer-year
        ``time`` dimension. Cross-year seasons at the record edges are
        computed from the available (partial) months."""
        sel = self.sel(obj)
        years = self.years(sel["time"])
        weights = sel["time"].dt.days_in_month.assign_coords(season_year=years)
        sel = sel.assign_coords(season_year=years)
        mean = (sel * weights).groupby("season_year").sum() / weights.groupby("season_year").sum()
        return mean.rename(season_year="time")


class SeasonalDefinition:
    """Ordered collection of Seasons. Iterate to visit each Season, index by
    name, or call ``mean``/``std`` to stack statistics on a ``season`` dim."""

    def __init__(self, *seasons: Season):
        self._seasons = {s.name: s for s in seasons}

    @classmethod
    def from_months(cls, months_by_name: dict) -> "SeasonalDefinition":
        return cls(*(Season(name, tuple(months)) for name, months in months_by_name.items()))

    @property
    def names(self) -> list[str]:
        return list(self._seasons)

    def __iter__(self):
        return iter(self._seasons.values())

    def __len__(self) -> int:
        return len(self._seasons)

    def __getitem__(self, name: str) -> Season:
        return self._seasons[name]

    def __contains__(self, name: str) -> bool:
        return name in self._seasons

    def __add__(self, other) -> "SeasonalDefinition":
        added = other if isinstance(other, SeasonalDefinition) else [other]
        return SeasonalDefinition(*self, *added)

    def subset(self, names: list[str]) -> "SeasonalDefinition":
        return SeasonalDefinition(*(self[name] for name in names))

    def _stack(self, results):
        return xr.concat(results, dim="season").assign_coords(season=self.names)

    def mean(self, obj):
        """Day-weighted climatological mean per season, on a ``season`` dim."""
        return self._stack([season.mean(obj) for season in self])

    def std(self, obj):
        """Day-weighted climatological std per season, on a ``season`` dim."""
        return self._stack([season.std(obj) for season in self])


NDJFM = Season("NDJFM", (11, 12, 1, 2, 3))

CVDP_SEASONS = SeasonalDefinition.from_months({
    "DJF": (12, 1, 2),
    "JFM": (1, 2, 3),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "JAS": (7, 8, 9),
    "SON": (9, 10, 11),
    "ANN": tuple(range(1, 13)),
})
