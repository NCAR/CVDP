# Call using 'from cvdp.vis import WinkelTripel'
from cartopy.crs import _WarpedRectangularProjection, Globe, WGS84_SEMIMAJOR_AXIS

class WinkelTripel(_WarpedRectangularProjection):
    """
    A Winkel-Tripel projection.
    Compromise modified azimuthal projection that is less distorted
    and more area-accurate. It is comparable to the Robinson
    projection in both appearance and popularity.
    The National Geographic Society uses the Winkel-Tripel projection
    for most of the maps they produce.
    """

    def __init__(self, central_longitude=0.0, central_latitude=0.0, globe=None):
        globe = globe or Globe(semimajor_axis=WGS84_SEMIMAJOR_AXIS)
        proj4_params = [('proj', 'wintri'),
                        ('lon_0', central_longitude),
                        ('lat_0', central_latitude)]
    
        super(WinkelTripel, self).__init__(proj4_params, central_longitude, globe=globe)
    

    @property
    def threshold(self):
        return 1e4