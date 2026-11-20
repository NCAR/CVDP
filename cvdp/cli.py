#!/usr/bin/env python
"""Command line interface for CVDP: ``cvdp run CONFIG OUT_DIR``."""

import argparse
import sys


def run(argv=None):
    """``cvdp run CONFIG OUT_DIR``: metrics to ``OUT_DIR/netcdf/``, plots and web pages to ``OUT_DIR``."""
    import xarray as xr

    from cvdp.compute import compute
    from cvdp.config import load_config
    from cvdp.products import products
    from cvdp.vis.driver import plot, write_webpages

    parser = argparse.ArgumentParser(prog="cvdp run", description="Compute CVDP metrics and plot them.")
    parser.add_argument("config", help="run configuration YAML (see cvdp/config.py)")
    parser.add_argument("out_dir", help="output directory")
    args = parser.parse_args(argv)
    paths = compute(load_config(args.config), f"{args.out_dir}/netcdf")
    plot(products([xr.open_dataset(p) for p in paths]), args.out_dir)
    write_webpages(args.out_dir)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] != ["run"]:
        sys.exit("usage: cvdp run CONFIG OUT_DIR")
    return run(argv[1:])


if __name__ == '__main__':
    main()
