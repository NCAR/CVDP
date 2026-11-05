from pathlib import Path

import pytest

pytest.importorskip("cartopy")
pytest.importorskip("jinja2")

import matplotlib.pyplot as plt  # noqa: E402

import cvdp.vis.driver as driver  # noqa: E402
from cvdp import cli  # noqa: E402

BASELINE_PNGS = Path("cvdp/test-data/legacy_baseline_pngs.txt")  # names in data/legacy_baseline/


def test_run_on_test_data_writes_the_legacy_png_set(tmp_path, monkeypatch):
    # Real config -> compute -> products -> file names; drawing is stubbed (test_vis_driver draws).
    monkeypatch.setattr(driver, "_figure", lambda *args: plt.figure())
    monkeypatch.setattr(driver, "compute_diff", lambda sim, ref: sim)
    cli.main(["run", "examples/test_data.yaml", str(tmp_path)])
    assert sorted(f.name for f in tmp_path.glob("*.png")) == BASELINE_PNGS.read_text().split()
    assert (tmp_path / "index.html").is_file() and (tmp_path / "index_indmem.html").is_file()
    assert sorted(f.name for f in (tmp_path / "netcdf").glob("*.nc")) == [
        "CESM2.001.pr.nc", "CESM2.001.psl.nc", "CESM2.001.tas.nc", "CESM2.001.ts.nc",
        "ERA5.psl.nc", "ERSSTv5.ts.nc", "GISTEMPv4.tas.nc", "GPCP.pr.nc"]
