# HRSG Chemical-Dosing DMC Benchmark

This repository contains the offline simulation code, controller-comparison settings, benchmark parameters, and generated result data for the controller-comparison experiments reported in the manuscript.

## Scope

The benchmark is an offline theoretical simulation of an HRSG chemical-dosing control problem. It does not contain field DCS measurements, Aspen DMC3 configuration files, plant operating records, customer documents, or manuscript drafts.

## Contents

- `src/run_robust_hrsg_simulation.py`: main deterministic benchmark generator and controller-comparison implementation.
- `src/run_pid_cross_fairness.py`: independent PID-pairing fairness check.
- `src/run_dmc_without_dv_ablation.py`: measurable-DV ablation experiment.
- `src/run_isolated_disturbance_events.py`: isolated disturbance-event companion experiment.
- `src/run_constraint_verification.py`: prediction-domain hard-constraint audit.
- `outputs/offline_simulation/`: generated CSV/JSON/TXT benchmark parameters, controller-comparison metrics, event records, and summary result data for reviewer inspection.

Figure-rendering scripts and manuscript-drafting files are not included.

## Environment

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproduce or Regenerate the Main Benchmark

```bash
python src/run_robust_hrsg_simulation.py
```

The main script uses `MASTER_SEED = 20260811` and writes deterministic local outputs to `outputs/offline_simulation/`. The generated CSV/JSON/TXT outputs currently included in this repository were produced from the same benchmark scripts.

## Optional Companion Experiments

Run these after the main benchmark has generated `outputs/offline_simulation/`.

```bash
python src/run_pid_cross_fairness.py
python src/run_dmc_without_dv_ablation.py
python src/run_isolated_disturbance_events.py
python src/run_constraint_verification.py
```

## Benchmark Parameters

The principal benchmark parameters are defined directly in `src/run_robust_hrsg_simulation.py`, including:

- sampling time `T_s = 1 min`;
- nominal model length `N = 60`;
- selected prediction horizon `P = 30`;
- control horizon `M = 5`;
- 3 CVs, 2 MVs, and 5 measured DVs;
- MV amplitude and rate constraints;
- CV operating bands;
- PI baseline tuning and DMC weighting matrices;
- Monte Carlo mismatch and noise settings.

The generated files in `outputs/offline_simulation/` provide a compact snapshot of the benchmark configuration and summary results.

## Evidence Boundary

All results are offline theoretical simulation results. They should not be interpreted as field DCS measurements, Aspen DMC3 deployment parameters, actual chemical-consumption records, interlock statistics, or operator-intervention records.
