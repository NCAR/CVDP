#!/usr/bin/env python
"""
utils.py

Utility functions used throughout the CVDP code base.
"""
from time import time
from importlib.metadata import version
import datetime


def log(msg: str):
    print(msg)


def get_time_stamp():
    return datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M')


def get_version():
    return version('cvdp')