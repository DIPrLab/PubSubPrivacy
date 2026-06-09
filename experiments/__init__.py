"""Per-experiment modules for the paper (each runnable as `python -m experiments.<name>`).

The heavy logic lives in ``experiments/engine.py`` (the core library); these
modules are focused, independently-runnable, cluster-shardable entry points.
``run_experiment.py`` at the repo root is a thin CLI over the engine.
"""
