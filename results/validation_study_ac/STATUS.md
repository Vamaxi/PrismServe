# Incomplete diagnostic AC attempt

This series stopped during `bursty_constrained / joint / seed 31` because the live worker reported that already-admitted work no longer had a feasible measured plan. It is excluded from successful benchmark aggregates. The original manifest remains `in_progress`; no failed run was labeled successful.

Raw telemetry for the aborted run records available host RAM around 0.8–1.0 GiB. Replaying the same input with `scripts/diagnose_admission.py` completed while minimum sampled host availability exceeded 2 GiB. This suggests a possible pressure-related confound but does not prove the original cause: the old worker cleared state without preserving its exact failing reservation.

The final implementation adds failure-only snapshots with active work, sampled capacities and current-plan estimates, and saves failed replay records before propagating an error. A new complete AC-conditioned series belongs to `results/validation_study_final/`. Its frozen source differs in failure instrumentation; serving policy and batch computation are retained. This diagnostic attempt and the earlier mixed-power complete study remain inspectable.
