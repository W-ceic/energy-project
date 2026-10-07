"""Fixed-parameter measurable-DV ablation for the existing 30 test scenarios.

This script does not change the original simulator, the 30 scenarios, or any
hold/trend result.  It creates only a DMC-without-DV baseline: the true plant
continues to receive its generated true-DV sequence, while the controller-side
FOPDT predictor receives the nominal zero DV vector for both prediction and
internal-model state advancement.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_robust_hrsg_simulation as base


ROOT = Path(__file__).resolve().parents[1]
BASE_OUT = ROOT / "outputs" / "offline_simulation"
OUT = ROOT / "outputs" / "dmc_without_dv_ablation"
OUT.mkdir(parents=True, exist_ok=True)
CONTROLLER = "DMC-without-DV"
HOLD = "DMC-DV hold"
TEST_IDS = tuple(range(1, 31))


def make_test_scenarios() -> List[base.Scenario]:
    """Recreate the pre-specified 30 Monte Carlo scenarios exactly by seed."""
    return [
        base.create_scenario(
            sid,
            base.MASTER_SEED + sid,
            360,
            True,
            True,
            "reference: ±15% K, ±20% τ, ±2 min θ",
        )
        for sid in TEST_IDS
    ]


def run_without_dv(scn: base.Scenario, pidpars: Dict[str, float]) -> Tuple[pd.DataFrame, List[Dict]]:
    """Run the existing constrained DMC with controller DV fixed at nominal zero.

    The only intentionally different information path is DV availability.  The
    controller receives the zero nominal DV vector in both ``step`` and
    ``advance``.  The true plant still receives ``dtrue`` on every simulation
    period, and CV feedback remains the measured-CV signal.
    """
    plant = base.FOPDTPlant(scn.Kp, scn.thetap, scn.taup)
    dmc = base.DMCController("hold", base.PIDController(pidpars))
    d_reference = np.zeros(base.ND)
    rows: List[Dict] = []

    for k in range(scn.T):
        ytrue = plant.output() + scn.w[k]
        ymeas = ytrue + scn.y_noise[k]
        dtrue = scn.dv_true[k]
        dmeas = dtrue + scn.dv_meas_noise[k]

        # No measured-DV information enters the controller or its nominal model.
        u, status, pred, active, dhat = dmc.step(ymeas, d_reference)
        feasible = dmc.last_qp_feasible

        row = {
            "scenario_id": scn.scenario_id,
            "random_seed": scn.seed,
            "scenario_label": scn.label,
            "time_min": k,
            "controller": CONTROLLER,
            "controller_DV_mode": "nominal_zero_no_measured_DV",
            "DMC_active": int(active),
            "QP_feasible": int(feasible),
            "QP_attempted": int(dmc.last_qp_attempted),
            "constraint_rate_active": int(status["rate"]),
            "constraint_amplitude_active": int(status["amp"]),
            "constraint_cv_active": int(status["cv"]),
        }
        for i, name in enumerate(base.CV_NAMES):
            row[f"{name}_true"] = ytrue[i]
            row[f"{name}_measured"] = ymeas[i]
            row[f"{name}_unmeasured_dist"] = scn.w[k, i]
            row[f"{name}_measurement_noise"] = scn.y_noise[k, i]
        for j, name in enumerate(base.MV_NAMES):
            row[name] = u[j]
        for j, name in enumerate(base.DV_NAMES):
            # The DCS-like trace is retained for audit; only its controller use is disabled.
            row[f"{name}_true"] = dtrue[j]
            row[f"{name}_measured"] = dmeas[j]
            row[f"{name}_predicted_1step"] = dhat[0, j]
        rows.append(row)

        # The internal model also advances with the nominal reference, not dmeas.
        dmc.advance(u, d_reference)
        # The true plant always retains the original true-DV excitation.
        plant.update(u, dtrue)

    return pd.DataFrame(rows), []


def verify_shared_exogenous_inputs(scenarios: List[base.Scenario], hold_ts: pd.DataFrame, stored_parameters: pd.DataFrame) -> pd.DataFrame:
    """Audit that regenerated scenarios reproduce all controller-exogenous traces."""
    checks = []
    for scn in scenarios:
        frame = hold_ts[hold_ts.scenario_id == scn.scenario_id].sort_values("time_min")
        if len(frame) != scn.T:
            raise RuntimeError(f"Scenario {scn.scenario_id}: stored hold trace length differs from scenario length")
        ok = int(frame.random_seed.nunique() == 1 and int(frame.random_seed.iloc[0]) == scn.seed)
        max_abs = 0.0
        for i, cv in enumerate(base.CV_NAMES):
            max_abs = max(max_abs, float(np.max(np.abs(frame[f"{cv}_unmeasured_dist"].to_numpy() - scn.w[:, i]))))
            max_abs = max(max_abs, float(np.max(np.abs(frame[f"{cv}_measurement_noise"].to_numpy() - scn.y_noise[:, i]))))
        for j, dv in enumerate(base.DV_NAMES):
            max_abs = max(max_abs, float(np.max(np.abs(frame[f"{dv}_true"].to_numpy() - scn.dv_true[:, j]))))
            max_abs = max(max_abs, float(np.max(np.abs(frame[f"{dv}_measured"].to_numpy() - (scn.dv_true[:, j] + scn.dv_meas_noise[:, j])))))
        parameter_rows = stored_parameters[
            (stored_parameters["scenario_id"] == scn.scenario_id)
            & (stored_parameters["CV"].isin(base.CV_NAMES))
            & (stored_parameters["input"].isin(base.MV_NAMES + base.DV_NAMES))
        ]
        if len(parameter_rows) != base.NY * (base.NU + base.ND):
            raise RuntimeError(f"Scenario {scn.scenario_id}: stored parameter record is incomplete")
        parameter_difference = 0.0
        for _, record in parameter_rows.iterrows():
            iy = base.CV_NAMES.index(record["CV"])
            j = (base.MV_NAMES + base.DV_NAMES).index(record["input"])
            parameter_difference = max(parameter_difference, abs(float(record["K_p"]) - scn.Kp[iy, j]))
            parameter_difference = max(parameter_difference, abs(float(record["theta_p_min"]) - scn.thetap[iy, j]))
            parameter_difference = max(parameter_difference, abs(float(record["tau_p_min"]) - scn.taup[iy, j]))
        checks.append({
            "scenario_id": scn.scenario_id,
            "random_seed": scn.seed,
            "same_true_plant": bool(parameter_difference < 1e-12),
            "same_initial_condition": True,
            "same_DV_events": True,
            "same_unmeasured_AR1": True,
            "same_CV_noise": True,
            "same_DV_noise": True,
            "stored_hold_seed_match": bool(ok),
            "max_abs_true_plant_parameter_difference": parameter_difference,
            "max_abs_regenerated_exogenous_trace_difference": max_abs,
            "audit_pass": bool(ok and parameter_difference < 1e-12 and max_abs < 1e-12),
        })
    audit = pd.DataFrame(checks)
    if not audit.audit_pass.all():
        raise RuntimeError("Existing hold traces do not match the regenerated 30-scenario exogenous inputs")
    return audit


def pair_metrics(no_dv: pd.DataFrame, hold: pd.DataFrame) -> pd.DataFrame:
    """Return fixed paired scenario statistics: hold minus DMC-without-DV."""
    result = base.make_pair_summary(
        no_dv,
        hold,
        CONTROLLER,
        HOLD,
        base.MASTER_SEED + 700,
    )
    result.insert(1, "contrast", "DMC-DV hold − DMC-without-DV")
    result["metric_manuscript_label"] = result.metric.replace({
        "IAE_MP_pH": "IAE_IP_pH",
        "RMSE_MP_pH": "RMSE_IP_pH",
        "Sigma_MP_pH": "Sigma_IP_pH",
        "PeakDev_MP_pH": "PeakDev_IP_pH",
        "OutRange_MP_pH": "OutRange_IP_pH",
    })
    return result


def scenario_metrics_with_total_mv_tv(frame: pd.DataFrame, scn: base.Scenario) -> Dict:
    """Keep original metrics and add the requested total MV total variation."""
    metrics = base.scenario_metrics(frame, scn)
    metrics["TV_total"] = metrics[f"TV_{base.MV_NAMES[0]}"] + metrics[f"TV_{base.MV_NAMES[1]}"]
    return metrics


def draw_representative(no_dv_ts: pd.DataFrame, hold_ts: pd.DataFrame, scenario_id: int = 7) -> None:
    """Plot the pre-specified scenario_id=7 without selecting a better case."""
    nd = no_dv_ts[no_dv_ts.scenario_id == scenario_id].sort_values("time_min")
    hd = hold_ts[hold_ts.scenario_id == scenario_id].sort_values("time_min")
    if len(nd) == 0 or len(hd) == 0:
        raise RuntimeError(f"Representative scenario_id={scenario_id} is unavailable")

    plt.rcParams.update({"font.size": 8, "font.family": "Arial", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.2), sharex=True)
    for dv in ("Cond_Flow", "Makeup_Flow"):
        axes[0].plot(hd.time_min, hd[f"{dv}_measured"], lw=0.8, label=f"{dv.replace('_', ' ')} measured")
    axes[0].set_ylabel("Measured DV deviation")
    axes[0].legend(frameon=False, ncol=2, fontsize=7)

    for cv, sp, band, ls in (("HP_pH", base.YSP[0], base.YMAX[0] - base.YMIN[0], "-"), ("MP_pH", base.YSP[1], base.YMAX[1] - base.YMIN[1], "--")):
        axes[1].plot(nd.time_min, (nd[f"{cv}_true"] - sp) / band, color="#666666", alpha=0.8, lw=0.8, ls=ls, label=f"{cv.replace('MP', 'IP')} without DV")
        axes[1].plot(hd.time_min, (hd[f"{cv}_true"] - sp) / band, color="#D55E00", alpha=0.8, lw=0.8, ls=ls, label=f"{cv.replace('MP', 'IP')} DV hold")
    axes[1].set_ylabel("Normalized pH deviation")
    axes[1].legend(frameon=False, ncol=2, fontsize=6)

    for mv in base.MV_NAMES:
        axes[2].plot(nd.time_min, nd[mv], color="#666666", alpha=0.8, lw=0.8, label=f"{mv.replace('_PID_SP', '')} without DV")
        axes[2].plot(hd.time_min, hd[mv], color="#D55E00", alpha=0.8, lw=0.8, label=f"{mv.replace('_PID_SP', '')} DV hold")
    axes[2].set_ylabel("MV deviation")
    axes[2].set_xlabel("Time (min)")
    axes[2].legend(frameon=False, ncol=2, fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "Fig_without_DV_vs_hold_scenario_7.png", dpi=350)
    plt.close(fig)


def write_report(no_dv_metrics: pd.DataFrame, comparison: pd.DataFrame, audit: pd.DataFrame, manifest: Dict) -> None:
    useful = [
        "IAE_HP_pH", "IAE_MP_pH", "RMSE_HP_pH", "RMSE_MP_pH",
        "Sigma_HP_pH", "Sigma_MP_pH", "PeakDev_HP_pH", "PeakDev_MP_pH",
        f"TV_{base.MV_NAMES[0]}", f"TV_{base.MV_NAMES[1]}", "TV_total",
        f"Saturation_{base.MV_NAMES[0]}", f"Saturation_{base.MV_NAMES[1]}",
        "QP_infeasible_count", "DMC_active_ratio",
    ]
    excerpt = comparison[comparison.metric.isin(useful)].copy()
    excerpt["metric"] = excerpt.metric.replace({"MP_pH": "IP_pH"}, regex=True)
    lines = [
        "# DMC-without-DV measurable-DV ablation report",
        "",
        "This is an offline theoretical simulation output. It does not modify the original 30 scenarios, DMC parameters, PID fallback parameters, or existing DMC-DV hold/trend results.",
        "",
        "## Fixed baseline definition",
        "",
        "The true plant continues to receive the original true-DV trajectory at every sampling period. The DMC-without-DV controller retains measured-CV feedback, unity innovation correction, P=40, M=5, Q=diag(4,4,0), R_u=diag(1.0,1.2), lambda_kappa=2.0, all MV/CV constraints, QP solver, and fallback logic. Its controller-side DV vector is fixed to the nominal zero vector for both future free-response prediction and internal-model state update.",
        "",
        "## Pairing audit",
        "",
        f"All {len(audit)} scenario_id pairs passed the regenerated exogenous-trace audit. Maximum absolute stored-vs-regenerated trace difference: {audit.max_abs_regenerated_exogenous_trace_difference.max():.3e}.",
        "",
        "## Main contrast",
        "",
        "The CSV `dmc_without_dv_vs_hold_pairwise_metrics.csv` reports DMC-DV hold minus DMC-without-DV, with scenario_id as the paired unit and 10,000 percentile-bootstrap 95% confidence intervals. For metrics where lower values are preferred, negative signed changes are in favor of the hold controller.",
        "",
        excerpt.to_csv(index=False),
        "",
        "Code variable `MP_pH` is displayed as `IP_pH` only in this report's manuscript-oriented labels; raw CSV column names remain unchanged for traceability.",
        "",
        "No conductivity slack trajectory is available because the original controller time-series logger does not record the internal QP slack vector. Zone and constraint-activity statistics that are already logged are retained.",
    ]
    (OUT / "dmc_without_dv_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "dmc_without_dv_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    source_manifest = json.loads((BASE_OUT / "simulation_manifest.json").read_text(encoding="utf-8"))
    if source_manifest["P"] != 40 or source_manifest["M"] != 5:
        raise RuntimeError("The existing benchmark manifest does not contain the fixed P=40, M=5 controller")
    base.configure_horizon(40)
    pidpars = source_manifest["PID_parameters"]

    all_ts = pd.read_csv(BASE_OUT / "monte_carlo_timeseries.csv")
    stored_parameters = pd.read_csv(BASE_OUT / "monte_carlo_scenario_parameters.csv")
    hold_ts = all_ts[all_ts.controller == HOLD].copy().sort_values(["scenario_id", "time_min"])
    if set(hold_ts.scenario_id.unique()) != set(TEST_IDS):
        raise RuntimeError("The existing hold output does not contain exactly scenario_id=1,...,30")

    scenarios = make_test_scenarios()
    audit = verify_shared_exogenous_inputs(scenarios, hold_ts, stored_parameters)

    no_dv_frames = []
    no_dv_metrics = []
    for scenario in scenarios:
        frame, _ = run_without_dv(scenario, pidpars)
        no_dv_frames.append(frame)
        no_dv_metrics.append(scenario_metrics_with_total_mv_tv(frame, scenario))
    no_dv_ts = pd.concat(no_dv_frames, ignore_index=True)
    no_dv_metrics_df = pd.DataFrame(no_dv_metrics).sort_values("scenario_id")

    hold_metrics = []
    for scenario in scenarios:
        frame = hold_ts[hold_ts.scenario_id == scenario.scenario_id].sort_values("time_min")
        hold_metrics.append(scenario_metrics_with_total_mv_tv(frame, scenario))
    hold_metrics_df = pd.DataFrame(hold_metrics).sort_values("scenario_id")
    comparison = pair_metrics(no_dv_metrics_df, hold_metrics_df)

    manifest = {
        "analysis_type": "fixed-parameter measurable-DV information ablation",
        "controllers_compared": [CONTROLLER, HOLD],
        "test_scenarios": "existing scenario_id=1,...,30; 360 min; 1 min sampling",
        "pairing": "same true plant, channel mismatch, initial condition, DV events, unmeasured AR(1) disturbance, CV noise, DV noise, and random seed within each scenario_id",
        "true_plant_DV": "unchanged: original d_true feeds the true plant on every sampling period",
        "without_DV_controller_input": "nominal zero DV vector for future prediction and internal-model update; measured DV is unavailable to the controller",
        "fixed_parameters": {"P": 40, "M": 5, "Q": "diag(4,4,0)", "R_u": "diag(1.0,1.2)", "lambda_kappa": 2.0},
        "unchanged_components": "measured CV feedback, unity innovation correction, MV/CV constraints, QP solver, fallback logic, and PID fallback parameters",
        "re_tuning_performed": False,
        "existing_hold_or_trend_results_modified": False,
        "bootstrap": "10,000 percentile resamples using the existing hold/trend bootstrap seed convention (MASTER_SEED+700); scenario_id is the resampling unit; signed paired difference = DMC-DV hold − DMC-without-DV",
    }

    audit.to_csv(OUT / "scenario_pairing_audit.csv", index=False)
    no_dv_ts.to_csv(OUT / "dmc_without_dv_timeseries.csv", index=False)
    no_dv_metrics_df.to_csv(OUT / "dmc_without_dv_metrics_per_scenario.csv", index=False)
    no_dv_metrics_df.to_csv(OUT / "dmc_without_dv_scenario_metrics.csv", index=False)
    hold_metrics_df.to_csv(OUT / "existing_hold_metrics_recomputed_from_original_timeseries.csv", index=False)
    comparison.to_csv(OUT / "dmc_without_dv_vs_hold_pairwise_metrics.csv", index=False)
    comparison.to_csv(OUT / "dmc_without_dv_vs_hold_paired_summary.csv", index=False)
    draw_representative(no_dv_ts, hold_ts, scenario_id=7)
    write_report(no_dv_metrics_df, comparison, audit, manifest)

    print(json.dumps({
        "output_dir": str(OUT),
        "pairing_audit_passed": int(audit.audit_pass.sum()),
        "n_scenarios": len(no_dv_metrics_df),
        "comparison_rows": len(comparison),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
