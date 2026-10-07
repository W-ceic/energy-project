# -*- coding: utf-8 -*-
"""RGA diagnostic and independent cross-pairing PID fairness study.

This script imports the current benchmark without changing its DMC model,
FOPDT constants, or the 30 final test-scenario definitions.  It creates only a
new independently tuned PID-cross baseline and writes separate output files.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import run_robust_hrsg_simulation as sim

OUT = sim.OUT
K_PH_MV = sim.KM[:2, :2]
RGA = K_PH_MV * np.linalg.inv(K_PH_MV).T
COND_K = float(np.linalg.cond(K_PH_MV))

class PIDCrossController(sim.PIDController):
    """Same position-form PI and anti-windup; cross RGA-guided pairings.

    MV1 (feedwater NH3) regulates IP pH and MV2 (condensate NH3) regulates
    HP pH.  All rate and amplitude limits are inherited unchanged.
    """
    def step(self, y):
        e = np.array([sim.YSP[1] - y[1], sim.YSP[0] - y[0]])
        raw = np.array([self.kp1, self.kp2]) * e + self.i
        lower = np.maximum(sim.UMIN, self.u + sim.DUMIN)
        upper = np.minimum(sim.UMAX, self.u + sim.DUMAX)
        limited = np.minimum(np.maximum(raw, lower), upper)
        ki = np.array([self.ki1, self.ki2])
        self.i += ki * e * sim.TS + self.kaw * (limited - raw)
        previous_u = self.u.copy()
        self.u = limited
        return self.u.copy(), {
            "rate": bool(np.any(np.isclose(np.abs(limited - previous_u), 0.03, atol=1e-8))),
            "amp": bool(np.any(np.isclose(np.abs(limited), 0.40, atol=1e-8))),
            "cv": False,
        }

def run_cross(scn, pars):
    plant = sim.FOPDTPlant(scn.Kp, scn.thetap, scn.taup)
    pid = PIDCrossController(pars)
    rows = []
    for k in range(scn.T):
        ytrue = plant.output() + scn.w[k]
        ymeas = ytrue + scn.y_noise[k]
        dtrue = scn.dv_true[k]
        dmeas = dtrue + scn.dv_meas_noise[k]
        u, status = pid.step(ymeas)
        row = {
            "scenario_id": scn.scenario_id, "random_seed": scn.seed,
            "scenario_label": scn.label, "time_min": k, "controller": "PID-cross",
            "DMC_active": 1, "QP_feasible": 1, "QP_attempted": 0,
            "constraint_rate_active": int(status["rate"]),
            "constraint_amplitude_active": int(status["amp"]), "constraint_cv_active": 0,
        }
        for i, name in enumerate(sim.CV_NAMES):
            row[f"{name}_true"] = ytrue[i]
            row[f"{name}_measured"] = ymeas[i]
            row[f"{name}_unmeasured_dist"] = scn.w[k, i]
            row[f"{name}_measurement_noise"] = scn.y_noise[k, i]
        for j, name in enumerate(sim.MV_NAMES):
            row[name] = u[j]
        for j, name in enumerate(sim.DV_NAMES):
            row[f"{name}_true"] = dtrue[j]
            row[f"{name}_measured"] = dmeas[j]
            row[f"{name}_predicted_1step"] = np.nan
        rows.append(row)
        plant.update(u, dtrue)
    frame = pd.DataFrame(rows)
    return frame

def score_cross(frame):
    m = sim.scenario_metrics(frame, sim.Scenario(0, 0, len(frame), sim.KM, sim.THETAM, sim.TAUM,
        np.zeros((len(frame), sim.ND)), np.zeros((len(frame), sim.ND)),
        np.zeros((len(frame), sim.NY)), np.zeros((len(frame), sim.NY)), [], False, "cross tuning"))
    iae = sum(m[f"IAE_{name}"] for name in sim.CV_NAMES)
    sat = m[f"Saturation_{sim.MV_NAMES[0]}"] + m[f"Saturation_{sim.MV_NAMES[1]}"]
    tv = m[f"TV_{sim.MV_NAMES[0]}"] + m[f"TV_{sim.MV_NAMES[1]}"]
    return iae + 75.0 * sat + 0.015 * tv

def tune_cross():
    # Independent held-out tuning scenario; neither the original PID tuning
    # seed nor the 30 final test scenario seeds are used.
    cal = sim.create_scenario(900, sim.MASTER_SEED + 177, T=240, mismatch=True,
                              with_noise=True, label="PID-cross tuning only")
    best = None
    for kp1 in (3.0, 5.0, 7.0, 9.0):
        for ki1 in (0.04, 0.08, 0.12):
            for kp2 in (3.0, 5.0, 7.0, 9.0, 11.0):
                for ki2 in (0.04, 0.08, 0.12):
                    pars = {"kp1": kp1, "ki1": ki1, "kp2": kp2, "ki2": ki2, "kaw": 0.5}
                    frame = run_cross(cal, pars)
                    score = score_cross(frame)
                    sat_max = max(np.mean(np.isclose(np.abs(frame[sim.MV_NAMES[0]]), .4)),
                                  np.mean(np.isclose(np.abs(frame[sim.MV_NAMES[1]]), .4)))
                    if sat_max > .10:
                        score += 1e3 * sat_max
                    if best is None or score < best[0]:
                        best = (score, pars)
    return best[1]

def main():
    # Use the already selected DMC horizon from the current experiment package.
    # No horizon selection is re-run and no DMC setting is changed.
    param = pd.read_csv(OUT / "simulation_parameters.csv")
    selected_p = int(float(param.loc[param.parameter == "P", "value"].iloc[0]))
    sim.configure_horizon(selected_p)
    diagonal_pars = sim.tune_pid()
    cross_pars = tune_cross()
    scenarios = [sim.create_scenario(i, sim.MASTER_SEED + i, 360, True, True,
                                     "reference: ±15% K, ±20% τ, ±2 min θ")
                 for i in range(1, 31)]
    diagonal_metrics, cross_metrics, dmc_metrics = [], [], []
    cross_ts = []
    for scn in scenarios:
        diagonal_frame, _, _ = sim.run_one(scn, "PID", diagonal_pars)
        cross_frame = run_cross(scn, cross_pars)
        dmc_frame, _, _ = sim.run_one(scn, "DMC-DV hold", diagonal_pars, "hold")
        diagonal_metrics.append(sim.scenario_metrics(diagonal_frame, scn))
        cross_metrics.append(sim.scenario_metrics(cross_frame, scn))
        dmc_metrics.append(sim.scenario_metrics(dmc_frame, scn))
        cross_ts.append(cross_frame)
    diagonal = pd.DataFrame(diagonal_metrics)
    cross = pd.DataFrame(cross_metrics)
    dmc = pd.DataFrame(dmc_metrics)
    combined = pd.concat([diagonal, cross, dmc], ignore_index=True)
    summary = pd.concat([
        sim.make_pair_summary(diagonal, cross, "PID-diagonal", "PID-cross", sim.MASTER_SEED + 800).assign(comparison_group="PID pairing"),
        sim.make_pair_summary(diagonal, dmc, "PID-diagonal", "DMC-DV hold", sim.MASTER_SEED + 810).assign(comparison_group="DMC vs diagonal"),
        sim.make_pair_summary(cross, dmc, "PID-cross", "DMC-DV hold", sim.MASTER_SEED + 820).assign(comparison_group="DMC vs cross"),
    ], ignore_index=True)
    rga = pd.DataFrame({
        "output": ["HP pH", "IP pH"], "MV1_FW_NH3": RGA[:, 0], "MV2_COND_NH3": RGA[:, 1],
        "K_MV1": K_PH_MV[:, 0], "K_MV2": K_PH_MV[:, 1], "cond_K": [COND_K, COND_K],
    })
    rga.to_csv(OUT / "pid_rga_diagnostic.csv", index=False)
    combined.to_csv(OUT / "pid_diagonal_cross_dmc_metrics_per_scenario.csv", index=False)
    summary.to_csv(OUT / "pid_diagonal_cross_dmc_pairwise_summary.csv", index=False)
    pd.concat(cross_ts, ignore_index=True).to_csv(OUT / "pid_cross_timeseries.csv", index=False)
    manifest = {
        "K_pH_MV": K_PH_MV.tolist(), "RGA": RGA.tolist(), "condition_number": COND_K,
        "PID_diagonal_pairing": "MV1→HP pH; MV2→IP pH",
        "PID_cross_pairing": "MV1→IP pH; MV2→HP pH",
        "PID_diagonal_parameters": diagonal_pars, "PID_cross_parameters": cross_pars,
        "same_implementation": "position-form PI with rate/amplitude saturation and back-calculation anti-windup",
        "same_constraints": {"u_bounds": [-.40, .40], "du_bounds": [-.03, .03], "Kaw": .5},
        "tuning": "independent 240 min scenario, seed=MASTER_SEED+177; same grid and score as PID-diagonal",
        "test_scenarios": "same 30 final scenarios as current DMC/PID-diagonal; identical objects, DV events, unmeasured disturbances, noise and seeds",
        "DMC_horizon_reused": selected_p,
    }
    (OUT / "pid_cross_fairness_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"RGA": RGA.tolist(), "condition_number": COND_K,
                      "pid_diagonal": diagonal_pars, "pid_cross": cross_pars,
                      "summary_file": str(OUT / 'pid_diagonal_cross_dmc_pairwise_summary.csv')}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
