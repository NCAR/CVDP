import matplotlib.pyplot as plt
import xarray

def plot_seasonal_ensemble_means(ensemble_avgs):
    plt.style.use("cvdp/visualization/cvdp.mplstyle")
    
    f = plt.figure()
    gridspec = f.add_gridspec(2, 2, height_ratios=[2, 2])
    ax1 = f.add_subplot(gridspec[0, 0])
    ax2 = f.add_subplot(gridspec[0, 1])
    ax3 = f.add_subplot(gridspec[1, 0])
    ax4 = f.add_subplot(gridspec[1, 1])
    
    ensemble_avgs.sel(season='DJF').plot(ax=ax1)
    ensemble_avgs.sel(season='MAM').plot(ax=ax2)
    ensemble_avgs.sel(season='JJA').plot(ax=ax3)
    ensemble_avgs.sel(season='SON').plot(ax=ax4)
    
    ax1.set_title(f"'{ensemble_avgs.name}' DJF Ensemble Mean")
    ax2.set_title(f"'{ensemble_avgs.name}' MAM Ensemble Mean")
    ax3.set_title(f"'{ensemble_avgs.name}' JJA Ensemble Mean")
    ax4.set_title(f"'{ensemble_avgs.name}' SON Ensemble Mean")
    return f