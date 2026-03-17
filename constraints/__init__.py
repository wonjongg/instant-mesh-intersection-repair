#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Geometric constraints for mesh optimization.

This module provides constraint functions for preserving geometric properties
during mesh intersection repair, including volume, area, and curvature constraints.
"""

from .volume import *
from .area import *
from .curvature import *
