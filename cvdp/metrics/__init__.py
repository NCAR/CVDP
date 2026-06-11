"""
cvdp.metrics

One module per metric category. Each module exposes stub functions with
documented signatures; implementations are added per-function.
"""
from . import climatology
from . import trends
from . import enso
from . import pacific_modes
from . import atlantic_modes
from . import atmospheric_modes
from . import ocean_circulation
from . import regional_timeseries
from . import seasons
