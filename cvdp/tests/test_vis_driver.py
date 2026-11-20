import pytest

pytest.importorskip("cartopy")
pytest.importorskip("jinja2")

from cvdp.products import products  # noqa: E402
from cvdp.tests.test_products import _output  # noqa: E402
from cvdp.vis.driver import plot, write_webpages  # noqa: E402


def test_three_layouts_per_product_and_web_pages(tmp_path):
    outputs = [_output("M", "psl", 1.0, member=".001."), _output("M", "psl", 2.0, member=".002."),
               _output("OBS", "psl", 0.0, reference=True)]
    plot([p for p in products(outputs) if p.name in ("psl_trends_djf", "nam_pattern_djf")], tmp_path)
    write_webpages(tmp_path)
    assert sorted(f.name for f in tmp_path.glob("*.png")) == sorted(
        [f"{n}.{layout}.png" for n in ("psl_trends_djf", "nam_pattern_djf") for layout in ("summary", "indmem", "indmemdiff")]
        + ["cas_cvdp-le_banner.png", "file_not_found_image.png"])
    page = (tmp_path / "index.html").read_text()
    assert "'psl_trends_djf.summary.png'" in page or '"psl_trends_djf.summary.png"' in page
    assert (tmp_path / "index_indmem.html").is_file()
