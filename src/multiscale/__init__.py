"""Multiscale dengue-rate prediction package.

Trains one frozen MLP per spatial unit at four nested spatial scales
(city, district, sector, SKATER region) under a leave-one-epidemic-year
cross-validation, using the exact same egg/dengue aggregation as the
SKATER pipeline.  Produces a spatial-scale vs prediction-accuracy
trade-off comparison.

Public entry points:
  src.multiscale.baselines  — city + district + sector models.
  src.multiscale.skater_cv  — 3-fold in-process SKATER + per-region models.
"""
