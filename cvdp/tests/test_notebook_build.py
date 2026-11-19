import pytest

nbformat = pytest.importorskip("nbformat")

from cvdp.vis.notebook_build import CVDPNotebook  # noqa: E402


def test_save_notebook_writes_version_header(tmp_path):
    CVDPNotebook().save_notebook(tmp_path / "out.ipynb")
    header = "".join(nbformat.read(tmp_path / "out.ipynb", as_version=4).cells[0].source)
    assert "Version: CVDP 0.1" in header and "Generation Timestamp: " in header
