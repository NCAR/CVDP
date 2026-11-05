"""
Plot driver: plot products (:mod:`cvdp.products`) -> PNGs in the three legacy
layouts, ``{product}.{summary,indmem,indmemdiff}.png`` as the legacy
baseline, and the web pages from the existing templates. Replaces
``vis/AtmOcnGR.graphics``, ``cvdp_utils.utils.gather_data`` and ``cvdp_utils.web``.
"""
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from cvdp.definitions import PATH_BANNER_PNG, PATH_ROOT_DIR, PATH_VARIABLE_DEFAULTS  # noqa: E402
from cvdp.vis.global_plots import (global_ensemble_plot, global_indmem_latlon_plot,  # noqa: E402
                                   global_indmemdiff_latlon_plot)
from cvdp.vis.polar_plots import (polar_ensemble_plot, polar_indmem_latlon_plot,  # noqa: E402
                                  polar_indmemdiff_latlon_plot)
from cvdp.vis.vis_utils import compute_diff  # noqa: E402

TITLES = {"summary": "Ensemble Summary: {} ({})", "indmem": "{} ({})\n", "indmemdiff": "{} Differences ({})\n"}


def plot(products, plot_dir):
    """Every product in every layout."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    settings = yaml.safe_load(Path(PATH_VARIABLE_DEFAULTS).read_text())
    for p in products:
        vres = settings[p.key][p.ptype]
        diffs = [compute_diff(s, r) for s in p.sims for r in p.refs]  # every member minus every reference
        for layout, title in TITLES.items():
            fig = _figure(p, layout, vres, diffs, title.format(p.title, p.season))
            fig.savefig(plot_dir / f"{p.name}.{layout}.png", bbox_inches="tight", dpi=150)
            plt.close(fig)


def _figure(p, layout, vres, diffs, title):
    """As legacy ``plot_dispatch``. Fresh lists: the plot functions extend them."""
    if layout == "summary":
        arrs = [list(p.sims_ens), list(p.refs_ens)]
        if p.map_type == "global":
            return global_ensemble_plot(arrs, list(diffs), p.vn, p.ptype, vres, title)
        return polar_ensemble_plot(arrs, list(diffs), p.vn, p.ptype, vres, title, p.key)
    if layout == "indmem":
        arrs = [list(p.sims), list(p.refs)]
        if p.map_type == "global":
            return global_indmem_latlon_plot(p.vn, arrs, vres, title, p.ptype)
        return polar_indmem_latlon_plot(p.vn, p.key, arrs, vres, title, p.ptype)
    if p.map_type == "global":
        return global_indmemdiff_latlon_plot(p.vn, list(diffs), vres, title, p.ptype)
    return polar_indmemdiff_latlon_plot(p.vn, p.key, list(diffs), vres, title, p.ptype)


def write_webpages(plot_dir):
    """``index.html`` and ``index_indmem.html``; images missing from ``plot_dir`` show a placeholder."""
    from jinja2 import Template

    plot_dir = Path(plot_dir)
    for f in (PATH_BANNER_PNG, Path(PATH_ROOT_DIR) / "file_not_found_image.png"):
        shutil.copy2(f, plot_dir)
    context = {
        "title": "Diagnostics Plots",
        "img": lambda f: f if (plot_dir / f).exists() else "file_not_found_image.png",
        "has_img": lambda f: (plot_dir / f).exists(),
        "create_time": datetime.now(timezone.utc),
    }
    for template, page in (("template_main.html", "index.html"), ("template_indmem.html", "index_indmem.html")):
        html = Template((Path(PATH_ROOT_DIR) / template).read_text()).render(context)
        (plot_dir / page).write_text(html)
