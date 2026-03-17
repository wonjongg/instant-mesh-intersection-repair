#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Energy functions for mesh intersection repair.

This module provides various energy formulations for penalizing mesh self-intersections,
including Triangle Proximity Energy (TPE), signed TPE variants, point-to-plane distance,
and conical penetration distance.
"""

from .distance import *
from .TPE import *
from .conical import *
