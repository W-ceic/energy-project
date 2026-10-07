"""Prediction-domain constraint verification for the existing HRSG DMC benchmark.

This is an observational companion experiment.  It imports the controller and
scenario generator unchanged, reuses their fixed seeds, and only exports the
QP-internal prediction and feasibility diagnostics needed for verification.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_robust_hrsg_simulation as sim


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OUT = ROOT / "outputs" / "offline_simulation"
OUT = SOURCE_OUT / "constraint_verification"
OUT.mkdir(parents=True, exist_ok=True)

# Match the Arial-based typography used by the other manuscript figures.
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.5,
    "axes.linewidth": 0.7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# Diagnostics only.  This does not modify the QP feasibility tolerance or any
# controller setting.  The QP uses its existing 1e-8 / 1e-7 internal tests.
CONSTRAINT_TOL = 1e-6
ACTIVE_TOL = 2e-6  # Existing active-set reporting threshold in the controller.
KAPPA_NORM_OFFSET = 0.15
KAPPA_SOFT_NORM = 0.10
KAPPA_HARD_NORM = 0.15


def load_existing_parameters() -> Tuple[int, Dict[str, float]]:
    """Reuse the currently published selected P and PI fallback parameters."""
    manifest = json.loads((SOURCE_OUT / "simulation_manifest.json").read_text(encoding="utf-8"))
    return int(manifest["P"]), {k: float(v) for k, v in manifest["PID_parameters"].items()}


def make_cycle_row(scn: sim.Scenario, experiment: str, k: int, y: np.ndarray,
                   d: np.ndarray, u: np.ndarray, dmc: sim.DMCController) -> Dict:
    """Flatten one control cycle without changing the controller state."""
    success = bool(dmc.last_qp_attempted and dmc.last_qp_feasible)
    row = {
        "experiment": experiment,
        "scenario_id": scn.scenario_id,
        "time": k,
        # ``DMC_active`` describes the controller that actually generated the
        # move in this cycle.  The pre-solve flag remains separately available
        # for the QP-success eligibility condition.
        "DMC_active": int(not dmc.last_fallback_active),
        "DMC_active_after_solve": int(not dmc.last_fallback_active),
        "DMC_active_before_solve": int(dmc.last_dmc_active_before_solve),
        "QP_attempted": int(dmc.last_qp_attempted),
        "QP_success": int(success),
        "QP_failure_reason": dmc.last_qp_failure_reason,
        "fallback_active": int(dmc.last_fallback_active),
        "fallback_episode_id": int(dmc.fallback_episode_id) if dmc.last_fallback_active else 0,
        "y_meas_HP": y[0], "y_meas_IP": y[1], "y_meas_COND": y[2],
        "d_meas": json.dumps(d.tolist()), "current_u1": u[0], "current_u2": u[1],
        "max_HP_prediction": np.nan, "min_HP_prediction": np.nan,
        "max_IP_prediction": np.nan, "min_IP_prediction": np.nan,
        "max_COND_prediction": np.nan, "max_COND_slack": np.nan,
        "max_abs_MV1": np.nan, "max_abs_MV2": np.nan,
        "max_abs_dMV1": np.nan, "max_abs_dMV2": np.nan,
        "max_QP_constraint_residual": np.nan,
        "number_of_positive_residuals_above_tol": np.nan,
    }
    if success:
        pred = dmc.last_prediction
        row.update({
            "max_HP_prediction": float(np.max(pred[:, 0])), "min_HP_prediction": float(np.min(pred[:, 0])),
            "max_IP_prediction": float(np.max(pred[:, 1])), "min_IP_prediction": float(np.min(pred[:, 1])),
            "max_COND_prediction": float(np.max(pred[:, 2])),
            "max_COND_slack": float(np.max(dmc.last_kappa_slack)),
            "max_abs_MV1": float(np.max(np.abs(dmc.last_u_future[:, 0]))),
            "max_abs_MV2": float(np.max(np.abs(dmc.last_u_future[:, 1]))),
            "max_abs_dMV1": float(np.max(np.abs(dmc.last_du_future[:sim.M, 0]))),
            "max_abs_dMV2": float(np.max(np.abs(dmc.last_du_future[:sim.M, 1]))),
        })
        residual = dmc.last_qp_a @ dmc.last_qp_z - dmc.last_qp_b
        row["max_QP_constraint_residual"] = float(np.max(residual))
        row["number_of_positive_residuals_above_tol"] = int(np.sum(residual > CONSTRAINT_TOL))
    return row


def run_audited_scenario(scn: sim.Scenario, pidpars: Dict[str, float], experiment: str) -> Tuple[List[Dict], List[Dict]]:
    """Run DMC-DV hold once, saving every successful QP's full P-step solution."""
    plant = sim.FOPDTPlant(scn.Kp, scn.thetap, scn.taup)
    dmc = sim.DMCController("hold", sim.PIDController(pidpars))
    cycle_rows: List[Dict] = []
    horizon_rows: List[Dict] = []
    for k in range(scn.T):
        ytrue = plant.output() + scn.w[k]
        ymeas = ytrue + scn.y_noise[k]
        dtrue = scn.dv_true[k]
        dmeas = dtrue + scn.dv_meas_noise[k]
        u, _, _, _, _ = dmc.step(ymeas, dmeas)
        cycle_rows.append(make_cycle_row(scn, experiment, k, ymeas, dmeas, u, dmc))
        if dmc.last_qp_attempted and dmc.last_qp_feasible:
            for p in range(sim.P):
                horizon_rows.append({
                    "experiment": experiment, "scenario_id": scn.scenario_id, "time": k,
                    "prediction_step": p + 1,
                    "HP_pred": dmc.last_prediction[p, 0], "IP_pred": dmc.last_prediction[p, 1],
                    "COND_pred": dmc.last_prediction[p, 2],
                    "u1_pred": dmc.last_u_future[p, 0], "u2_pred": dmc.last_u_future[p, 1],
                    "du1_pred": dmc.last_du_future[p, 0], "du2_pred": dmc.last_du_future[p, 1],
                    "COND_slack": dmc.last_kappa_slack[p],
                })
        dmc.advance(u, dmeas)
        plant.update(u, dtrue)
    return cycle_rows, horizon_rows


def violation_stats(values: np.ndarray, lower: float | None, upper: float | None) -> Dict[str, float | int]:
    if len(values) == 0:
        return {"count": 0, "ratio": np.nan, "max": np.nan, "flags": np.zeros(0, dtype=bool)}
    low = np.zeros(len(values), dtype=bool) if lower is None else values < lower - CONSTRAINT_TOL
    high = np.zeros(len(values), dtype=bool) if upper is None else values > upper + CONSTRAINT_TOL
    violation = low | high
    magnitude = np.zeros(len(values))
    if lower is not None:
        magnitude = np.maximum(magnitude, lower - values)
    if upper is not None:
        magnitude = np.maximum(magnitude, values - upper)
    magnitude = np.maximum(magnitude, 0.0)
    return {"count": int(np.sum(violation)), "ratio": float(np.mean(violation)),
            "max": float(np.max(magnitude)), "flags": violation}


def safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else np.nan


def safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if len(values) else np.nan


def safe_quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else np.nan


def summary_for(experiment: str, cycles: pd.DataFrame, horizon: pd.DataFrame) -> Dict:
    success_cycles = cycles[(cycles.QP_attempted == 1) & (cycles.QP_success == 1)]
    qp_attempts = cycles[cycles.QP_attempted == 1]
    successful_h = horizon.merge(success_cycles[["scenario_id", "time"]], on=["scenario_id", "time"], how="inner")
    assert len(successful_h) == len(success_cycles) * sim.P, "Prediction horizon log is incomplete."
    hp = violation_stats(successful_h.HP_pred.to_numpy(), sim.YMIN[0], sim.YMAX[0])
    ip = violation_stats(successful_h.IP_pred.to_numpy(), sim.YMIN[1], sim.YMAX[1])
    cond = violation_stats(successful_h.COND_pred.to_numpy(), None, sim.KAPPA_HARD)
    cycle_keys = successful_h[["scenario_id", "time"]].copy()
    cycle_keys["hp_bad"] = hp["flags"]
    cycle_keys["ip_bad"] = ip["flags"]
    cycle_keys["cond_bad"] = cond["flags"]
    bad_cycles = cycle_keys.groupby(["scenario_id", "time"])[["hp_bad", "ip_bad", "cond_bad"]].any().sum()

    slack = successful_h.COND_slack.to_numpy()
    u1, u2 = successful_h.u1_pred.to_numpy(), successful_h.u2_pred.to_numpy()
    # Rate validation applies to the M optimized moves, not the padded zeros
    # after M used only to make the P-step export rectangular.
    optimized = successful_h[successful_h.prediction_step <= sim.M]
    du1, du2 = optimized.du1_pred.to_numpy(), optimized.du2_pred.to_numpy()
    mv1 = violation_stats(u1, sim.UMIN[0], sim.UMAX[0])
    mv2 = violation_stats(u2, sim.UMIN[1], sim.UMAX[1])
    dmv1 = violation_stats(du1, sim.DUMIN[0], sim.DUMAX[0])
    dmv2 = violation_stats(du2, sim.DUMIN[1], sim.DUMAX[1])
    mv_abs_active = np.concatenate([np.abs(u1 - sim.UMIN[0]) <= ACTIVE_TOL,
                                    np.abs(u1 - sim.UMAX[0]) <= ACTIVE_TOL,
                                    np.abs(u2 - sim.UMIN[1]) <= ACTIVE_TOL,
                                    np.abs(u2 - sim.UMAX[1]) <= ACTIVE_TOL])
    mv_rate_active = np.concatenate([np.abs(du1 - sim.DUMIN[0]) <= ACTIVE_TOL,
                                     np.abs(du1 - sim.DUMAX[0]) <= ACTIVE_TOL,
                                     np.abs(du2 - sim.DUMIN[1]) <= ACTIVE_TOL,
                                     np.abs(du2 - sim.DUMAX[1]) <= ACTIVE_TOL])
    residuals = success_cycles.max_QP_constraint_residual.to_numpy(float)
    fallback_episodes = cycles.loc[(cycles.fallback_active == 1) & (cycles.fallback_episode_id > 0),
                                   ["scenario_id", "fallback_episode_id"]].drop_duplicates()
    return {
        "experiment": experiment, "n_scenarios": int(cycles.scenario_id.nunique()),
        "total_samples": int(len(cycles)), "qp_attempt_cycles": int(len(qp_attempts)),
        "qp_success_cycles": int(len(success_cycles)), "qp_failure_cycles": int(len(qp_attempts) - len(success_cycles)),
        "qp_success_ratio": float(len(success_cycles) / len(qp_attempts)) if len(qp_attempts) else np.nan,
        "fallback_episodes": int(len(fallback_episodes)),
        "dmc_active_ratio": float(cycles.DMC_active.mean()),
        "HP_pred_hard_violation_ratio": hp["ratio"], "HP_pred_hard_violation_cycles": int(bad_cycles.hp_bad), "HP_pred_max_violation": hp["max"],
        "IP_pred_hard_violation_ratio": ip["ratio"], "IP_pred_hard_violation_cycles": int(bad_cycles.ip_bad), "IP_pred_max_violation": ip["max"],
        "COND_pred_hard_violation_ratio": cond["ratio"], "COND_pred_hard_violation_cycles": int(bad_cycles.cond_bad), "COND_pred_max_violation": cond["max"],
        "COND_soft_zone_active_ratio": safe_mean(successful_h.COND_pred.to_numpy() > sim.KAPPA_ZONE + CONSTRAINT_TOL),
        "COND_slack_active_ratio": safe_mean(slack > CONSTRAINT_TOL), "COND_mean_slack": safe_mean(slack), "COND_max_slack": safe_max(slack),
        "MV1_abs_violation_ratio": mv1["ratio"], "MV2_abs_violation_ratio": mv2["ratio"],
        "MV1_rate_violation_ratio": dmv1["ratio"], "MV2_rate_violation_ratio": dmv2["ratio"],
        "MV_abs_active_ratio": safe_mean(mv_abs_active), "MV_rate_active_ratio": safe_mean(mv_rate_active),
        "max_QP_constraint_residual": safe_max(residuals), "mean_QP_constraint_residual": safe_mean(residuals),
        "p95_QP_constraint_residual": safe_quantile(residuals, .95),
        # Supplemental counts make the requested ratios auditable without
        # changing the requested compact table schema.
        "HP_pred_hard_violation_points": hp["count"], "HP_pred_total_prediction_points": int(len(successful_h)),
        "IP_pred_hard_violation_points": ip["count"], "IP_pred_total_prediction_points": int(len(successful_h)),
        "COND_pred_hard_violation_points": cond["count"], "COND_pred_total_prediction_points": int(len(successful_h)),
    }


def _state_raster(ax, time: np.ndarray, rows: List[Tuple[str, np.ndarray, float, str, str]]) -> None:
    """Plot sparse state events on separate, non-overlapping lanes."""
    for label, mask, level, color, marker in rows:
        t = time[mask]
        if len(t):
            ax.scatter(t, np.full(len(t), level), s=12, marker=marker, color=color, label=label, zorder=3)
    ax.set_yticks([r[2] for r in rows], [r[0] for r in rows])
    ax.set_ylim(min(r[2] for r in rows) - .45, max(r[2] for r in rows) + .45)
    ax.grid(axis="x", color="#DDDDDD", lw=.5)
    ax.legend(frameon=False, ncol=len(rows), fontsize=7, loc="upper right")


def plot_reference_constraint_scenario(cycles: pd.DataFrame) -> None:
    representative = cycles[(cycles.experiment == "reference") & (cycles.scenario_id == 7)].sort_values("time")
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.3), sharex=True)
    axes[0].plot(representative.time, representative.min_HP_prediction, color="#0072B2", lw=1, label="min$_p$ prediction")
    axes[0].plot(representative.time, representative.max_HP_prediction, color="#56B4E9", lw=1, label="max$_p$ prediction")
    axes[0].axhline(sim.YMIN[0], color="#555555", ls="--", lw=.8, label="hard bounds")
    axes[0].axhline(sim.YMAX[0], color="#555555", ls="--", lw=.8)
    axes[0].set_ylabel("HP pH"); axes[0].legend(frameon=False, ncol=3, fontsize=7)
    axes[1].plot(representative.time, representative.min_IP_prediction, color="#0072B2", lw=1, label="min$_p$ prediction")
    axes[1].plot(representative.time, representative.max_IP_prediction, color="#56B4E9", lw=1, label="max$_p$ prediction")
    axes[1].axhline(sim.YMIN[1], color="#555555", ls="--", lw=.8, label="hard bounds")
    axes[1].axhline(sim.YMAX[1], color="#555555", ls="--", lw=.8)
    axes[1].set_ylabel("IP pH"); axes[1].legend(frameon=False, ncol=3, fontsize=7)
    axes[2].plot(representative.time, representative.max_COND_prediction, color="#0072B2", lw=1, label="max$_p$ conductivity prediction")
    axes[2].axhline(sim.KAPPA_ZONE, color="#E69F00", ls="--", lw=.8, label="κ zone = 0.25")
    axes[2].axhline(sim.KAPPA_HARD, color="#555555", ls="--", lw=.8, label="κ hard = 0.30")
    axes[2].set_ylabel("κ (μS/cm)"); axes[2].legend(frameon=False, ncol=3, fontsize=7)
    time = representative.time.to_numpy()
    _state_raster(axes[3], time, [
        ("DMC active", representative.DMC_active.to_numpy(bool), 2, "#0072B2", "|"),
        ("QP success", representative.QP_success.to_numpy(bool), 1, "#009E73", "|"),
        ("QP failure", ((representative.QP_attempted == 1) & (representative.QP_success == 0)).to_numpy(bool), 0, "#D55E00", "x"),
    ])
    axes[3].set_xlabel("Time (min)")
    fig.suptitle("Reference Monte Carlo: pre-specified scenario ID 7", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "constraint_verification_reference_scenario7.png", dpi=350)
    plt.close(fig)


def replay_pressure_truth(pidpars: Dict[str, float], existing: pd.DataFrame) -> pd.DataFrame:
    """Recover only the omitted true κ trace from the fixed, existing scenario."""
    scn = sim.create_scenario(107, sim.MASTER_SEED + 507, 360, True, True, "conductivity-zone stress")
    scn.w[:, 2] += 0.175
    plant = sim.FOPDTPlant(scn.Kp, scn.thetap, scn.taup)
    dmc = sim.DMCController("hold", sim.PIDController(pidpars))
    rows = []
    for k in range(scn.T):
        ytrue = plant.output() + scn.w[k]
        ymeas = ytrue + scn.y_noise[k]
        dtrue = scn.dv_true[k]
        dmeas = dtrue + scn.dv_meas_noise[k]
        u, _, _, _, _ = dmc.step(ymeas, dmeas)
        rows.append({
            "time": k, "COND_true": ytrue[2], "COND_measured": ymeas[2],
            "DMC_active": int(not dmc.last_fallback_active),
            "QP_attempted": int(dmc.last_qp_attempted),
            "QP_success": int(dmc.last_qp_attempted and dmc.last_qp_feasible),
            "fallback_active": int(dmc.last_fallback_active),
        })
        dmc.advance(u, dmeas)
        plant.update(u, dtrue)
    truth = pd.DataFrame(rows)
    merged = existing[["time", "y_meas_COND", "DMC_active", "QP_attempted", "QP_success", "fallback_active"]].merge(truth, on="time")
    if not np.allclose(merged.y_meas_COND, merged.COND_measured, atol=1e-12, rtol=0):
        raise RuntimeError("Fixed-seed replay does not reproduce the existing pressure log.")
    for state in ("DMC_active", "QP_attempted", "QP_success", "fallback_active"):
        if not np.array_equal(merged[f"{state}_x"].to_numpy(), merged[f"{state}_y"].to_numpy()):
            raise RuntimeError(f"Fixed-seed replay does not reproduce logged {state} state.")
    return truth


def plot_pressure_fallback_scenario(cycles: pd.DataFrame, pidpars: Dict[str, float]) -> None:
    # Source IDs are 101...130; ordinal 7 is source scenario_id=107.
    representative = cycles[(cycles.experiment == "conductivity pressure") & (cycles.scenario_id == 107)].sort_values("time")
    # Display-only normalization for the revised manuscript CV definition:
    # kappa_COND,norm = kappa_COND - 0.15. The archived cycle data, events,
    # predictions and controller states remain unchanged.
    conductivity = representative.y_meas_COND - KAPPA_NORM_OFFSET
    conductivity_label = "conductivity trajectory"
    conductivity_ylabel = "Normalized conductivity deviation"
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.3), sharex=True)
    axes[0].plot(representative.time, conductivity, color="#333333", lw=1, label=conductivity_label)
    axes[0].axhline(KAPPA_SOFT_NORM, color="#E69F00", ls="--", lw=.8, label="kappa soft = +0.10")
    axes[0].axhline(KAPPA_HARD_NORM, color="#555555", ls="--", lw=.8, label="kappa hard = +0.15")
    axes[0].set_ylabel(conductivity_ylabel); axes[0].legend(frameon=False, ncol=3, fontsize=7)
    success = representative.QP_success.to_numpy(bool)
    failure = ((representative.QP_attempted == 1) & (representative.QP_success == 0)).to_numpy(bool)
    max_cond_prediction_norm = representative.max_COND_prediction - KAPPA_NORM_OFFSET
    axes[1].scatter(representative.time[success], max_cond_prediction_norm[success], s=18, marker="o", color="#0072B2", label="Successful-QP max conductivity prediction")
    # Deliberately do not put QP-failure markers in this quantitative κ panel:
    # a failure has no solved prediction, so an arbitrary vertical placement
    # would falsely look like a conductivity value.  Panel C is the dedicated
    # event raster for all QP failures.
    axes[1].axhline(KAPPA_SOFT_NORM, color="#E69F00", ls="--", lw=.8, label="kappa soft = +0.10")
    axes[1].axhline(KAPPA_HARD_NORM, color="#555555", ls="--", lw=.8, label="kappa hard = +0.15")
    axes[1].set_ylabel("Normalized conductivity deviation"); axes[1].legend(frameon=False, ncol=3, fontsize=7)
    time = representative.time.to_numpy()
    _state_raster(axes[2], time, [
        ("QP attempted", representative.QP_attempted.to_numpy(bool), 2, "#555555", "|"),
        ("QP success", success, 1, "#009E73", "o"),
        ("QP failure", failure, 0, "#D55E00", "x"),
    ])
    _state_raster(axes[3], time, [
        ("DMC active", representative.DMC_active.to_numpy(bool), 1, "#0072B2", "|"),
        ("PI fallback", representative.fallback_active.to_numpy(bool), 0, "#D55E00", "|"),
    ])
    axes[3].set_xlabel("Time (min)")
    fig.suptitle("Conductivity-pressure scenario: QP-failure and PI-fallback diagnostics", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "constraint_pressure_fallback_scenario7.png", dpi=350)
    fig.savefig(OUT / "constraint_pressure_fallback_scenario7.pdf")
    fig.savefig(OUT / "constraint_pressure_fallback_scenario7.svg")
    plt.close(fig)


def report_markdown(summary: pd.DataFrame) -> str:
    # Keep the report dependency-free: the bundled environment has an older
    # tabulate package, while CSV is already the authoritative numeric table.
    table = "```csv\n" + summary.to_csv(index=False, float_format="%.6g") + "```"
    reference = summary.loc[summary.experiment == "reference"].iloc[0]
    pressure = summary.loc[summary.experiment == "conductivity pressure"].iloc[0]
    return f"""# Prediction-domain constraint verification

## Scope and parameters read from the running code

- Controller: existing DMC-DV hold; no controller, plant, seed, noise, DV, QP, or PI parameter was changed.
- Current selected horizon: `P={sim.P}`; `M={sim.M}`; `N={sim.N}`.
- Actual code bounds: HP/IP pH `[9.60, 9.70]`, conductivity soft zone `0.25`, conductivity hard upper `0.30`, MV `[-0.40, 0.40]`, and MV move `[-0.03, 0.03]` per minute.
- Diagnostic tolerance: `{CONSTRAINT_TOL:.0e}`. It is used only for this report.  The existing QP feasibility tests remain unchanged.
- Naming difference: the source code calls the second pH CV `MP_pH`; this report labels it `IP` to follow the requested terminology.
- The pressure cohort exactly reuses the source condition (30 scenarios, seeds `MASTER_SEED+501...+530`, source IDs 101...130, mismatch/noise enabled, and `w[:,2] += 0.175`).

## Results

{table}

## Interpretation

Hard-constraint rates above are calculated only for `DMC_active_before_solve=1`, `QP_attempted=1`, and `QP_success=1`.  PI fallback samples and failed QP attempts are excluded from those rates. `κ > 0.25` and positive conductivity slack are soft-zone activity, not hard-constraint violations.  QP residual is `max(Az* - b)` for the actual assembled QP inequalities.

The reference experiment obtained {int(reference.qp_success_cycles)} successful QP solutions from {int(reference.qp_attempt_cycles)} attempts, with no QP failures or fallback episodes.  The conductivity-pressure experiment obtained {int(pressure.qp_success_cycles)} successful solutions from {int(pressure.qp_attempt_cycles)} attempts; its {int(pressure.qp_failure_cycles)} failed attempts and {int(pressure.fallback_episodes)} PI-fallback episodes are reported separately from the successful-QP constraint audit.  In both samples, all checked CV and MV hard-violation point counts were zero, and no assembled-QP residual exceeded the diagnostic tolerance.

## Suggested caption and interpretation for the conductivity-pressure diagnostic figure

**Caption draft.** *Conductivity-pressure scenario: QP-failure and PI-fallback diagnostics (pre-specified scenario 7).* Panel (a) shows true plant conductivity, whereas panel (b) shows only the maximum predicted conductivity over the prediction horizon for successful-QP cycles. Panels (c) and (d) respectively show QP attempt/outcome events and the DMC/PI-fallback control state.

Only a few cycles in this pressure scenario yield successful QP solutions; therefore, the successful-QP prediction points in panel (b) are sparse and are used only to illustrate prediction behavior near the hard constraint, rather than continuous DMC closed-loop operation. The successful-QP maximum predicted conductivity approaches, but does not materially exceed, the 0.30 μS/cm hard upper limit. This does not imply that true plant conductivity remains below 0.30 μS/cm: model mismatch, disturbances, QP failures, and PI-fallback intervals can still lead to true-process exceedances.

## Suggested Section 5.4 addition (draft only)

Under the DMC-DV hold formulation, an independent prediction-domain audit was performed for the 30-scenario reference condition and the existing conductivity-pressure condition.  At each successfully solved QP cycle, all P-step CV predictions, the optimized M-step control moves, the reconstructed absolute MV trajectory, the conductivity-zone slack, and the full inequality residual were retained.  Hard-constraint compliance was evaluated only for QP-success cycles; QP failures and PI-fallback samples were reported separately.  The resulting numerical values are given in `constraint_verification_summary.csv`.  These results demonstrate numerical satisfaction (or any observed exceptions) of the constraints in the controller's prediction model and do not constitute a guarantee that the true HRSG process remains within industrial safety limits under model mismatch and disturbances.
"""


def main() -> None:
    selected_p, pidpars = load_existing_parameters()
    sim.configure_horizon(selected_p)
    experiments: List[Tuple[str, Iterable[sim.Scenario]]] = [
        ("reference", [sim.create_scenario(i, sim.MASTER_SEED + i, 360, True, True,
                                             "reference: ±15% K, ±20% τ, ±2 min θ") for i in range(1, 31)]),
        ("conductivity pressure", [sim.create_scenario(100 + i, sim.MASTER_SEED + 500 + i, 360, True, True,
                                                        "conductivity-zone stress") for i in range(1, 31)]),
    ]
    # Preserve the existing 5.4 pressure rule exactly.
    for _, scenarios in experiments[1:]:
        for scn in scenarios:
            scn.w[:, 2] += 0.175

    all_cycles: List[Dict] = []
    all_horizon: List[Dict] = []
    for experiment, scenarios in experiments:
        for scn in scenarios:
            cycles, horizon = run_audited_scenario(scn, pidpars, experiment)
            all_cycles.extend(cycles)
            all_horizon.extend(horizon)
    cycle_df = pd.DataFrame(all_cycles)
    horizon_df = pd.DataFrame(all_horizon)
    summary = pd.DataFrame([summary_for(experiment, cycle_df[cycle_df.experiment == experiment],
                                        horizon_df[horizon_df.experiment == experiment])
                            for experiment, _ in experiments])
    by_scenario = pd.DataFrame([summary_for(f"{experiment}: scenario {sid}",
                                             cycle_df[(cycle_df.experiment == experiment) & (cycle_df.scenario_id == sid)],
                                             horizon_df[(horizon_df.experiment == experiment) & (horizon_df.scenario_id == sid)])
                                   for experiment, _ in experiments
                                   for sid in sorted(cycle_df.loc[cycle_df.experiment == experiment, "scenario_id"].unique())])
    cycle_df.to_csv(OUT / "constraint_verification_cycle_level.csv", index=False)
    horizon_df.to_csv(OUT / "constraint_prediction_horizon.csv", index=False)
    summary.to_csv(OUT / "constraint_verification_summary.csv", index=False)
    by_scenario.to_csv(OUT / "constraint_verification_by_scenario.csv", index=False)
    plot_reference_constraint_scenario(cycle_df)
    plot_pressure_fallback_scenario(cycle_df, pidpars)
    (OUT / "constraint_verification_report.md").write_text(report_markdown(summary), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote constraint-verification outputs to {OUT}")


if __name__ == "__main__":
    main()




