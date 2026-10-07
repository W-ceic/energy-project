"""Reproducible offline theoretical HRSG chemical-dosing simulation.

The script implements a 3-CV/2-MV/5-DV equivalent process.  It is deliberately
an offline theoretical simulation: no setting or numerical result represents a
field DCS measurement, Aspen DMC3 configuration, or chemical-consumption record.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from math import sqrt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "offline_simulation"
OUT.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 20260811
TS = 1.0
N = 60
# P is selected by the independent closed-loop horizon study in
# ``select_prediction_horizon``.  The default is kept at 30 so that a direct
# invocation of this script uses the selected setting after the study.
P = 30
M = 5
NY, NU, ND = 3, 2, 5
YSP = np.array([9.65, 9.65, 0.15])  # plant nominal operating point
YMIN = np.array([9.60, 9.60, 0.00])
YMAX = np.array([9.70, 9.70, 0.30])
KAPPA_ZONE = 0.25
KAPPA_HARD = 0.30
KAPPA_ZONE_WEIGHT = 2.0
UMIN = np.array([-0.40, -0.40])
UMAX = np.array([0.40, 0.40])
DUMIN = np.array([-0.03, -0.03])
DUMAX = np.array([0.03, 0.03])
# pH is set-point-tracking; conductivity is a one-sided zone-constrained CV.
# It is not assigned a fictitious conductivity setpoint.
Q = np.diag([4.0, 4.0, 0.0])
RU = np.diag([1.0, 1.2])
# Nominal output scaling used only to balance numerical units in the QP.
# It is one third of full-band normalization to avoid over-aggressive moves.
YSCALE = 1.0 / (3.0 * (YMAX - YMIN))
CV_NOISE = np.array([0.002, 0.002, 0.003])
DV_NOISE = np.array([0.0075, 0.0075, 0.0075, 0.005, 0.005])
W_STD = np.array([0.004, 0.004, 0.007])
AR = 0.95
TREND_L = 5
TREND_MAX = np.array([0.025, 0.025, 0.025, 0.035, 0.050])
PREDICTION_HORIZONS = (1, 5, 10, 20, 30, 40)

CV_NAMES = ["HP_pH", "MP_pH", "COND_kappa"]
MV_NAMES = ["FW_NH3_PID_SP", "COND_NH3_PID_SP"]
DV_NAMES = ["LP_Drum_pH", "LP_Drum_Cond", "LP_ECO_pH", "Cond_Flow", "Makeup_Flow"]
COLORS = {"PID": "#555555", "DMC-DV hold": "#D55E00", "Proposed DMC-DV trend": "#0072B2"}

# Nominal controller-internal FOPDT model. Values are theoretical simulation settings.
KM = np.array([
    [0.060, 0.015, 0.030, -0.025, 0.015, -0.030, -0.040],
    [0.050, 0.012, 0.020, -0.020, 0.010, -0.020, -0.030],
    [-0.006, -0.080, 0.000, 0.004, -0.003, 0.060, 0.080],
])
THETAM = np.array([
    [5, 7, 3, 2, 3, 4, 4],
    [6, 8, 3, 2, 3, 4, 4],
    [6, 4, 2, 2, 2, 3, 3],
], dtype=int)
TAUM = np.array([
    [18, 20, 12, 10, 12, 18, 15],
    [20, 22, 14, 12, 14, 18, 16],
    [16, 15, 8, 10, 10, 14, 12],
], dtype=float)

@dataclass
class Scenario:
    scenario_id: int
    seed: int
    T: int
    Kp: np.ndarray
    thetap: np.ndarray
    taup: np.ndarray
    dv_true: np.ndarray
    dv_meas_noise: np.ndarray
    w: np.ndarray
    y_noise: np.ndarray
    events: List[Dict]
    mismatch: bool
    label: str = "reference"

def step_response(K: float, theta: int, tau: float, length: int) -> np.ndarray:
    ans = np.zeros(length)
    for p in range(length):
        t = (p + 1) * TS
        if t > theta:
            ans[p] = K * (1.0 - np.exp(-(t - theta) / tau))
    return ans

def build_gu() -> np.ndarray:
    gu = np.zeros((P * NY, M * NU))
    for iy in range(NY):
        for iu in range(NU):
            s = step_response(KM[iy, iu], int(THETAM[iy, iu]), TAUM[iy, iu], P)
            for p in range(P):
                for m in range(M):
                    if p >= m:
                        gu[p * NY + iy, m * NU + iu] = s[p - m]
    return gu

def configure_horizon(prediction_horizon: int) -> None:
    """Rebuild lifted matrices after changing the prediction horizon."""
    global P, GU, QBAR, RUBAR, SCALEBAR, GUN, H_BASE, S_ACC, A_CV, A_QP
    P = int(prediction_horizon)
    GU = build_gu()
    QBAR = np.kron(np.eye(P), Q)
    RUBAR = np.kron(np.eye(M), RU)
    SCALEBAR = np.kron(np.eye(P), np.diag(YSCALE))
    GUN = SCALEBAR @ GU
    H_BASE = 2.0 * (GUN.T @ QBAR @ GUN + RUBAR)
    H_BASE += 1e-8 * np.eye(M * NU)
    S_ACC = np.zeros((M * NU, M * NU))
    for m in range(M):
        for j in range(NU):
            for q in range(m + 1):
                S_ACC[m * NU + j, q * NU + j] = 1.0
    # pH has two-sided hard bounds; conductivity has one predictive hard upper
    # constraint.  The softer κ-zone requirement is represented by explicit
    # non-negative slack variables inside the QP, not by max(·) postprocessing.
    rows = []
    for p in range(P):
        for iy in (0, 1):
            rows.extend([GU[p * NY + iy], -GU[p * NY + iy]])
        rows.append(GU[p * NY + 2])
    A_CV = np.vstack(rows)
    A_QP = np.vstack([np.eye(M * NU), -np.eye(M * NU), S_ACC, -S_ACC, A_CV])

configure_horizon(P)

def hard_qp_active_set(h: np.ndarray, f: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray | None, Dict[str, bool | str]]:
    """Solve a strictly convex hard-constrained QP by a primal active-set method.

    Phase I uses scipy.optimize.linprog only when zero moves are not feasible.
    The returned solution satisfies the QP constraints directly; it is not an
    unconstrained DMC solution followed by clipping.
    """
    n = len(f)
    x = np.zeros(n)
    if np.any(a @ x - b > 1e-7):
        phase = linprog(np.zeros(n), A_ub=a, b_ub=b, bounds=[(None, None)] * n, method="highs")
        if not phase.success:
            return None, {"rate": False, "amp": False, "cv": False,
                          "failure_reason": f"phase-I infeasible ({phase.message})"}
        x = phase.x
    slack = b - a @ x
    working = list(np.where(slack <= tol)[0])
    for _ in range(80):
        g = h @ x + f
        if working:
            aw = a[working, :]
            kkt = np.block([[h, aw.T], [aw, np.zeros((len(working), len(working)))]] )
            rhs = np.concatenate([-g, np.zeros(len(working))])
            sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
            p, lam = sol[:n], sol[n:]
        else:
            p = np.linalg.solve(h, -g)
            lam = np.empty(0)
        if np.max(np.abs(p)) < 1e-9:
            if len(lam) == 0 or np.all(lam >= -1e-8):
                s = b - a @ x
                return x, {"rate": bool(np.any(np.abs(s[:2*M*NU]) < 2e-6)),
                           "amp": bool(np.any(np.abs(s[2*M*NU:4*M*NU]) < 2e-6)),
                           "cv": bool(np.any(np.abs(s[4*M*NU:4*M*NU + 5*P]) < 2e-6)),
                           "failure_reason": ""}
            del working[int(np.argmin(lam))]
            continue
        alpha = 1.0
        entering = None
        for i in range(a.shape[0]):
            if i in working:
                continue
            ap = float(a[i] @ p)
            if ap > 1e-12:
                alpha_i = float((b[i] - a[i] @ x) / ap)
                if alpha_i < alpha:
                    alpha, entering = max(0.0, alpha_i), i
        x = x + alpha * p
        if entering is not None and entering not in working:
            working.append(entering)
    return None, {"rate": False, "amp": False, "cv": False,
                  "failure_reason": "active-set iteration limit"}

def dv_effect(dhat: np.ndarray, dcur: np.ndarray) -> np.ndarray:
    """Returns Gd ΔD contribution for P predicted samples."""
    eff = np.zeros((P, NY))
    increments = np.vstack([dhat[0] - dcur, np.diff(dhat, axis=0)])
    for iy in range(NY):
        for jd in range(ND):
            s = step_response(KM[iy, NU + jd], int(THETAM[iy, NU + jd]), TAUM[iy, NU + jd], P)
            for p in range(P):
                eff[p, iy] += sum(s[p - q] * increments[q, jd] for q in range(p + 1))
    return eff.reshape(-1)

def build_b(u_prev: np.ndarray, base_prediction: np.ndarray) -> np.ndarray:
    rate_upper = np.tile(DUMAX, M)
    rate_lower = -np.tile(DUMIN, M)
    urep = np.tile(u_prev, M)
    amp_upper = np.tile(UMAX, M) - urep
    amp_lower = -(np.tile(UMIN, M) - urep)
    cv = []
    for p in range(P):
        for iy in (0, 1):
            cv.extend([YMAX[iy] - base_prediction[p * NY + iy],
                       -(YMIN[iy] - base_prediction[p * NY + iy])])
        cv.append(KAPPA_HARD - base_prediction[p * NY + 2])
    return np.concatenate([rate_upper, rate_lower, amp_upper, amp_lower, np.asarray(cv)])

class PIDController:
    def __init__(self, pars: Dict[str, float], u0=None):
        self.kp1, self.ki1, self.kp2, self.ki2, self.kaw = (pars[k] for k in ("kp1", "ki1", "kp2", "ki2", "kaw"))
        self.i = np.zeros(2)
        self.u = np.zeros(2) if u0 is None else np.asarray(u0, float).copy()
    def step(self, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, bool]]:
        # Decentralized reference baseline: MV1–HP pH and MV2–IP pH are the
        # selected diagonal pairings.  Conductivity remains a monitored
        # constrained variable; it is not silently used to deactivate MV2.
        e = np.array([YSP[0] - y[0], YSP[1] - y[1]])
        raw = self.u * 0.0 + np.array([self.kp1, self.kp2]) * e + self.i
        lower = np.maximum(UMIN, self.u + DUMIN)
        upper = np.minimum(UMAX, self.u + DUMAX)
        limited = np.minimum(np.maximum(raw, lower), upper)
        ki = np.array([self.ki1, self.ki2])
        self.i += ki * e * TS + self.kaw * (limited - raw)
        previous_u = self.u.copy()
        self.u = limited
        return self.u.copy(), {"rate": bool(np.any(np.isclose(np.abs(limited - previous_u), 0.03, atol=1e-8))),
                               "amp": bool(np.any(np.isclose(np.abs(limited), 0.40, atol=1e-8))), "cv": False}

class DMCController:
    def __init__(self, mode: str, pid_fallback: PIDController):
        self.mode = mode
        self.u = np.zeros(NU)
        self.dhist: List[np.ndarray] = []
        self.infeas_streak = 0
        self.active = True
        self.cooldown = 0
        self.last_qp_attempted = False
        self.last_qp_feasible = True
        # Diagnostic state is observational only: it is populated for the
        # prediction-domain constraint-verification experiment and is never
        # read by the control law.
        self.last_qp_failure_reason = ""
        self.last_dmc_active_before_solve = True
        self.last_fallback_active = False
        self.fallback_episode_id = 0
        self.last_qp_a = None
        self.last_qp_b = None
        self.last_qp_z = None
        self.last_prediction = None
        self.last_u_future = None
        self.last_du_future = None
        self.last_kappa_slack = None
        self.pid_fallback = pid_fallback
        self.internal_model = FOPDTPlant(KM, THETAM, TAUM)
    def forecast(self, d: np.ndarray) -> np.ndarray:
        self.dhist.append(d.copy())
        if len(self.dhist) > TREND_L + 1:
            self.dhist.pop(0)
        if self.mode == "hold" or len(self.dhist) <= TREND_L:
            return np.tile(d, (P, 1))
        slope = (d - self.dhist[0]) / TREND_L
        slope = np.minimum(np.maximum(slope, -TREND_MAX), TREND_MAX)
        pred = np.array([d + (p + 1) * slope for p in range(P)])
        return np.minimum(np.maximum(pred, -0.50), 0.50)
    def step(self, y: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, Dict[str, bool], np.ndarray, bool]:
        self.last_dmc_active_before_solve = self.active
        self.last_qp_a = self.last_qp_b = self.last_qp_z = None
        self.last_prediction = self.last_u_future = self.last_du_future = self.last_kappa_slack = None
        self.last_qp_failure_reason = ""
        self.last_fallback_active = False
        if not self.active:
            u, st = self.pid_fallback.step(y)
            self.u = u
            self.last_qp_attempted = False
            self.last_qp_feasible = True
            self.last_fallback_active = True
            self.cooldown -= 1
            if self.cooldown <= 0:
                # Controlled bumpless re-entry is allowed only after a finite
                # PID fallback interval; the internal model has kept advancing.
                self.active = True
                self.infeas_streak = 0
            return u, st, np.tile(y, (P, 1)), False, np.tile(d, (P, 1))
        dhat = self.forecast(d)
        # Internal-model free response plus current innovation correction.
        innovation = y - self.internal_model.output()
        clone = self.internal_model.clone()
        fmat = np.zeros((P, NY))
        for p in range(P):
            clone.update(self.u, dhat[p])
            fmat[p] = clone.output() + innovation
        base = fmat.reshape(-1)
        # Standard QP decision z=[ΔU; ε_κ].  ε_κ is a non-negative zone slack
        # satisfying κ_hat−κ_zone≤ε_κ; κ_hat≤κ_hard is retained separately.
        # This closes the soft-zone/hard-upper formulation without max(·).
        tracking = base - np.tile(YSP, P)
        n_du = M * NU
        h = np.zeros((n_du + P, n_du + P))
        h[:n_du,:n_du] = H_BASE
        h[n_du:,n_du:] = 2.0 * KAPPA_ZONE_WEIGHT * np.eye(P)
        h += 1e-8 * np.eye(n_du + P)
        f = np.zeros(n_du + P)
        f[:n_du] = 2.0 * GUN.T @ QBAR @ (SCALEBAR @ tracking)
        base_a = np.hstack([A_QP, np.zeros((A_QP.shape[0], P))])
        kappa_rows = np.vstack([GU[p * NY + 2] for p in range(P)])
        zone_a = np.hstack([kappa_rows, -np.eye(P)])
        slack_a = np.hstack([np.zeros((P, n_du)), -np.eye(P)])
        a = np.vstack([base_a, zone_a, slack_a])
        b = np.concatenate([build_b(self.u, base),
                            np.full(P, KAPPA_ZONE) - base[2::NY],
                            np.zeros(P)])
        z, status = hard_qp_active_set(h, f, a, b)
        du = None if z is None else z[:n_du]
        self.last_qp_attempted = True
        self.last_qp_feasible = du is not None
        self.last_qp_failure_reason = str(status.get("failure_reason", ""))
        if du is None:
            self.infeas_streak += 1
            if self.infeas_streak >= 3:
                self.active = False
                self.cooldown = 30
                self.fallback_episode_id += 1
                self.last_fallback_active = True
                u, st = self.pid_fallback.step(y)
                self.u = u
                return u, st, base.reshape(P, NY), False, dhat
            return self.u.copy(), {"rate": False, "amp": False, "cv": False}, base.reshape(P, NY), True, dhat
        self.infeas_streak = 0
        u_prev = self.u.copy()
        self.u = self.u + du[:NU]
        pred = (base + GU @ du).reshape(P, NY)
        du_moves = du.reshape(M, NU)
        du_future = np.zeros((P, NU))
        du_future[:M] = du_moves
        u_future = np.empty((P, NU))
        u_running = u_prev.copy()
        for p in range(P):
            u_running = u_running + du_future[p]
            u_future[p] = u_running
        self.last_qp_a, self.last_qp_b, self.last_qp_z = a, b, z
        self.last_prediction = pred
        self.last_u_future, self.last_du_future = u_future, du_future
        self.last_kappa_slack = z[n_du:].copy()
        return self.u.copy(), status, pred, True, dhat
    def advance(self, u: np.ndarray, d: np.ndarray):
        self.internal_model.update(u, d)

class FOPDTPlant:
    def __init__(self, K: np.ndarray, theta: np.ndarray, tau: np.ndarray):
        self.K, self.theta, self.tau = K, theta.astype(int), tau
        self.state = np.zeros((NY, NU + ND))
        self.max_delay = int(np.max(theta))
        self.hist = [np.zeros(self.max_delay + 1) for _ in range(NU + ND)]
    def output(self) -> np.ndarray:
        return YSP + self.state.sum(axis=1)
    def clone(self):
        other = FOPDTPlant(self.K, self.theta, self.tau)
        other.state = self.state.copy()
        other.hist = [h.copy() for h in self.hist]
        return other
    def update(self, u: np.ndarray, d: np.ndarray):
        inp = np.concatenate([u, d])
        for j in range(NU + ND):
            self.hist[j] = np.concatenate([[inp[j]], self.hist[j][:-1]])
        for iy in range(NY):
            for j in range(NU + ND):
                a = np.exp(-TS / self.tau[iy, j])
                delayed = self.hist[j][int(self.theta[iy, j])]
                self.state[iy, j] = a * self.state[iy, j] + (1 - a) * self.K[iy, j] * delayed

def random_event(rng, dv_index, start, amplitude, kind, duration):
    return {"dv": DV_NAMES[dv_index], "dv_index": int(dv_index), "start_min": int(start), "amplitude": float(amplitude), "kind": kind, "duration_min": int(duration)}

def apply_event(arr: np.ndarray, event: Dict):
    j, st, amp, dur, kind = event["dv_index"], event["start_min"], event["amplitude"], event["duration_min"], event["kind"]
    end = min(len(arr), st + dur)
    if kind == "step":
        arr[st:, j] += amp
    elif kind == "ramp":
        arr[st:end, j] += np.linspace(0, amp, end-st, endpoint=True)
        arr[end:, j] += amp
    else:  # temporary pulse
        half = max(1, (end-st)//2)
        arr[st:st+half, j] += np.linspace(0, amp, half, endpoint=False)
        arr[st+half:end, j] += np.linspace(amp, 0, end-(st+half), endpoint=False)

def create_scenario(sid: int, seed: int, T: int = 360, mismatch=True, with_noise=True, label: str = "reference") -> Scenario:
    rng = np.random.default_rng(seed)
    if mismatch:
        kp = KM * (1 + rng.uniform(-0.15, 0.15, KM.shape))
        taup = TAUM * (1 + rng.uniform(-0.20, 0.20, TAUM.shape))
        thetap = np.maximum(0, THETAM + rng.choice(np.array([-2, -1, 0, 1, 2]), THETAM.shape))
    else:
        kp, taup, thetap = KM.copy(), TAUM.copy(), THETAM.copy()
    dv = np.zeros((T, ND))
    # Small independent background DV movement.
    for j in range(ND):
        for k in range(1, T):
            dv[k, j] = 0.85 * dv[k-1, j] + rng.normal(0, 0.004)
    events = []
    starts = sorted(rng.choice(np.arange(70, max(71, T-45)), size=3, replace=False))
    for idx, st in enumerate(starts):
        j = 3 if idx != 1 else 4
        lo, hi = ((0.08, 0.15) if j == 3 else (0.15, 0.25))
        ev = random_event(rng, j, int(st), float(rng.choice([-1,1])*rng.uniform(lo,hi)), str(rng.choice(["step","ramp","pulse"])), int(rng.integers(25, 55)))
        events.append(ev); apply_event(dv, ev)
    if T > 1000:
        # Long-horizon run: additional small events plus a few larger events.
        for idx, st in enumerate(range(480, T-120, 480)):
            j = 3 if idx % 2 == 0 else 4
            lo, hi = ((0.04, 0.10) if j == 3 else (0.08, 0.16))
            if idx % 9 == 0:
                lo, hi = ((0.12, 0.20) if j == 3 else (0.20, 0.30))
            ev = random_event(rng, j, int(st + rng.integers(-30,31)), float(rng.choice([-1,1])*rng.uniform(lo,hi)), str(rng.choice(["step","ramp","pulse"])), int(rng.integers(20,80)))
            events.append(ev); apply_event(dv, ev)
    # Controller-invisible colored water-chemistry disturbance.
    eta_std = W_STD * sqrt(1 - AR**2)
    w = np.zeros((T, NY))
    w[0] = rng.normal(0, W_STD * 0.6)
    for k in range(1, T):
        w[k] = AR*w[k-1] + rng.normal(0, eta_std)
    stage_start = int(rng.integers(150, max(151, T-60)))
    stage_duration = int(rng.integers(20, 50))
    stage = rng.normal([0.006, 0.006, 0.010], [0.002,0.002,0.003])
    w[stage_start:min(T, stage_start+stage_duration)] += stage
    events.append({"dv":"unmeasured_colored_disturbance", "dv_index":-1, "start_min":stage_start, "amplitude":float(np.linalg.norm(stage)), "kind":"enhanced_colored", "duration_min":stage_duration})
    ynoise = rng.normal(0, CV_NOISE, (T, NY)) if with_noise else np.zeros((T, NY))
    dnoise = rng.normal(0, DV_NOISE, (T, ND)) if with_noise else np.zeros((T, ND))
    return Scenario(sid, seed, T, kp, thetap, taup, dv, dnoise, w, ynoise, events, mismatch, label)

def run_one(scn: Scenario, controller_name: str, pidpars: Dict[str,float], mode: str | None = None) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    plant = FOPDTPlant(scn.Kp, scn.thetap, scn.taup)
    pid = PIDController(pidpars)
    dmc = None if controller_name == "PID" else DMCController(mode, PIDController(pidpars))
    rows, pred_rows = [], []
    for k in range(scn.T):
        ytrue = plant.output() + scn.w[k]
        ymeas = ytrue + scn.y_noise[k]
        dtrue = scn.dv_true[k]
        dmeas = dtrue + scn.dv_meas_noise[k]
        if controller_name == "PID":
            u, status = pid.step(ymeas)
            pred = np.full((P, NY), np.nan); active = True; dhat = np.full((P, ND), np.nan); feasible = True
        else:
            u, status, pred, active, dhat = dmc.step(ymeas, dmeas)
            feasible = dmc.last_qp_feasible
        row = {"scenario_id":scn.scenario_id, "random_seed":scn.seed, "scenario_label":scn.label, "time_min":k, "controller":controller_name,
               "DMC_active":int(active), "QP_feasible":int(feasible), "QP_attempted":int(dmc.last_qp_attempted) if dmc is not None else 0, "constraint_rate_active":int(status["rate"]),
               "constraint_amplitude_active":int(status["amp"]), "constraint_cv_active":int(status["cv"])}
        for i,n in enumerate(CV_NAMES):
            row[f"{n}_true"], row[f"{n}_measured"], row[f"{n}_unmeasured_dist"], row[f"{n}_measurement_noise"] = ytrue[i], ymeas[i], scn.w[k,i], scn.y_noise[k,i]
        for j,n in enumerate(MV_NAMES): row[n] = u[j]
        for j,n in enumerate(DV_NAMES):
            row[f"{n}_true"], row[f"{n}_measured"] = dtrue[j], dmeas[j]
            row[f"{n}_predicted_1step"] = np.nan if controller_name == "PID" else dhat[0,j]
        rows.append(row)
        if controller_name != "PID":
            for h in (q for q in PREDICTION_HORIZONS if q <= P):
                if k+h < scn.T:
                    for i,n in enumerate(CV_NAMES):
                        pred_rows.append({"scenario_id":scn.scenario_id,"random_seed":scn.seed,"condition":"mismatch_noise" if scn.mismatch else "nominal","horizon_step":h,"CV":n,"time_min":k,"predicted":pred[h-1,i],"measured_future":np.nan})
        if dmc is not None:
            dmc.advance(u, dmeas)
        plant.update(u, dtrue)
    frame = pd.DataFrame(rows)
    # Attach future measured values to prediction rows after the run.
    if pred_rows:
        for r in pred_rows:
            r["measured_future"] = float(frame.loc[frame.time_min == r["time_min"] + r["horizon_step"], f"{r['CV']}_measured"].iloc[0])
    return frame, pred_rows, {"scenario_id":scn.scenario_id,"random_seed":scn.seed,"events":scn.events}

def scenario_metrics(frame: pd.DataFrame, scn: Scenario) -> Dict:
    out = {"scenario_id":scn.scenario_id,"random_seed":scn.seed,"scenario_label":scn.label,"controller":frame.controller.iloc[0]}
    for i,n in enumerate(CV_NAMES):
        e = frame[f"{n}_true"].to_numpy() - YSP[i]
        out[f"IAE_{n}"] = float(np.sum(np.abs(e))*TS)
        out[f"RMSE_{n}"] = float(np.sqrt(np.mean(e**2)))
        out[f"Sigma_{n}"] = float(np.std(frame[f"{n}_true"],ddof=1))
        out[f"PeakDev_{n}"] = float(np.max(np.abs(e)))
        if n == "COND_kappa":
            out[f"OutRange_{n}"] = float(np.mean(frame[f"{n}_true"] > KAPPA_HARD))
            out[f"ZoneActive_{n}"] = float(np.mean(frame[f"{n}_true"] > KAPPA_ZONE))
        else:
            out[f"OutRange_{n}"] = float(np.mean((frame[f"{n}_true"] < YMIN[i]) | (frame[f"{n}_true"] > YMAX[i])))
    rec = []
    for ev in [e for e in scn.events if e["dv_index"] >= 0]:
        start = ev["start_min"]
        for i,n in enumerate(CV_NAMES):
            x = frame.loc[frame.time_min >= start, f"{n}_true"].to_numpy()
            found = None
            for q in range(max(0,len(x)-4)):
                inrange = x[q:q+5] <= KAPPA_HARD if n == "COND_kappa" else ((x[q:q+5] >= YMIN[i]) & (x[q:q+5] <= YMAX[i]))
                if np.all(inrange):
                    found = q; break
            rec.append(np.nan if found is None else float(found))
    out["RecoveryTime_mean_min"] = float(np.nanmean(rec)) if np.any(~np.isnan(rec)) else np.nan
    out["NotRecovered_count"] = int(np.sum(np.isnan(rec)))
    for j,n in enumerate(MV_NAMES):
        u = frame[n].to_numpy()
        out[f"TV_{n}"] = float(np.sum(np.abs(np.diff(u))))
        out[f"Saturation_{n}"] = float(np.mean(np.isclose(np.abs(u),0.4,atol=1e-6)))
    out["QP_infeasible_count"] = int(np.sum((frame.QP_attempted == 1) & (frame.QP_feasible == 0))) if frame.controller.iloc[0] != "PID" else 0
    out["DMC_withdrawal_count"] = int(np.sum((frame.DMC_active.shift(1,fill_value=1)==1)&(frame.DMC_active==0))) if frame.controller.iloc[0] != "PID" else 0
    out["DMC_active_ratio"] = float(frame.DMC_active.mean())
    out["MV_amplitude_constraint_active_ratio"] = float(frame.constraint_amplitude_active.mean())
    out["MV_rate_constraint_active_ratio"] = float(frame.constraint_rate_active.mean())
    out["CV_constraint_active_ratio"] = float(frame.constraint_cv_active.mean())
    return out

def pid_score(frame: pd.DataFrame) -> float:
    m = scenario_metrics(frame, Scenario(0,0,len(frame),KM,THETAM,TAUM,np.zeros((len(frame),ND)),np.zeros((len(frame),ND)),np.zeros((len(frame),NY)),np.zeros((len(frame),NY)),[],False,"tuning"))
    iae = sum(m[f"IAE_{n}"] for n in CV_NAMES)
    sat = m[f"Saturation_{MV_NAMES[0]}"] + m[f"Saturation_{MV_NAMES[1]}"]
    tv = m[f"TV_{MV_NAMES[0]}"] + m[f"TV_{MV_NAMES[1]}"]
    return iae + 75*sat + 0.015*tv

def tune_pid() -> Dict[str,float]:
    # Held-out tuning scenario: it never enters the 30 independent test
    # scenarios used for PID/DMC reporting.
    cal = create_scenario(0, MASTER_SEED+77, T=240, mismatch=True, with_noise=True, label="PID tuning only")
    best = None
    for kp1 in (3.0,5.0,7.0,9.0):
        for ki1 in (0.04,0.08,0.12):
            for kp2 in (3.0,5.0,7.0,9.0,11.0):
                for ki2 in (0.04,0.08,0.12):
                    pars={"kp1":kp1,"ki1":ki1,"kp2":kp2,"ki2":ki2,"kaw":0.5}
                    f,_,_=run_one(cal,"PID",pars)
                    score=pid_score(f)
                    sat=max(np.mean(np.isclose(np.abs(f[MV_NAMES[0]]),.4)),np.mean(np.isclose(np.abs(f[MV_NAMES[1]]),.4)))
                    if sat>0.10: score += 1e3*sat
                    if best is None or score < best[0]: best=(score,pars)
    return best[1]

def select_prediction_horizon(pidpars: Dict[str, float]) -> Tuple[int, pd.DataFrame]:
    """Select P on held-out scenarios using closed-loop, not R², criteria.

    Candidates use twelve independent validation scenarios.  Feasibility is
    ranked first; among feasible candidates the minimum mean HP/IP IAE is
    selected, with total MV TV used as a tie-breaker within 1% of that minimum.
    These scenarios are not included in PID tuning or in the 30 test scenarios.
    """
    rows = []
    validation = [create_scenario(500 + sid, MASTER_SEED + 6000 + sid, 360,
                                  True, True, "horizon-selection only")
                  for sid in range(1, 13)]
    for candidate in (10, 20, 30, 40):
        configure_horizon(candidate)
        records = []
        for scn in validation:
            frame, _, _ = run_one(scn, "DMC-DV hold", pidpars, "hold")
            records.append(scenario_metrics(frame, scn))
        result = pd.DataFrame(records)
        rows.append({"P": candidate, "M": M, "n_validation_scenarios": len(result),
                     "HP_IP_IAE_mean": float((result.IAE_HP_pH + result.IAE_MP_pH).mean()),
                     "TV_total_mean": float((result.TV_FW_NH3_PID_SP + result.TV_COND_NH3_PID_SP).mean()),
                     "QP_infeasible_total": int(result.QP_infeasible_count.sum()),
                     "DMC_active_ratio_mean": float(result.DMC_active_ratio.mean())})
    sweep = pd.DataFrame(rows)
    feasible = sweep[sweep.QP_infeasible_total == sweep.QP_infeasible_total.min()].copy()
    best_iae = feasible.HP_IP_IAE_mean.min()
    near_best = feasible[feasible.HP_IP_IAE_mean <= 1.01 * best_iae]
    selected = int(near_best.sort_values(["TV_total_mean", "P"]).iloc[0].P)
    sweep["selected"] = sweep.P == selected
    configure_horizon(selected)
    return selected, sweep

def summarize_prediction(records: List[Dict]) -> Tuple[pd.DataFrame,pd.DataFrame]:
    df = pd.DataFrame(records).dropna()
    rows=[]
    for (cond,sid,h,cv), g in df.groupby(["condition","scenario_id","horizon_step","CV"]):
        err=g.predicted-g.measured_future; den=float(YMAX[CV_NAMES.index(cv)]-YMIN[CV_NAMES.index(cv)])
        rows.append({"condition":cond,"scenario_id":sid,"horizon_step":h,"CV":cv,"RMSE":float(np.sqrt(np.mean(err**2))),"MAE":float(np.mean(np.abs(err))),"R2":float(1-np.sum(err**2)/np.sum((g.measured_future-g.measured_future.mean())**2)) if np.std(g.measured_future)>1e-12 else np.nan,"NRMSE_band_pct":float(np.sqrt(np.mean(err**2))/den*100)})
    per=pd.DataFrame(rows)
    summ=per.groupby(["condition","horizon_step","CV"],as_index=False).agg(RMSE_mean=("RMSE","mean"),RMSE_SD=("RMSE","std"),MAE_mean=("MAE","mean"),R2_mean=("R2","mean"),NRMSE_band_pct_mean=("NRMSE_band_pct","mean"))
    return per,summ

DV_TREND_BOOTSTRAP_SEED = MASTER_SEED + 700  # fixed base seed for Table 9 paired percentile bootstrap


def bootstrap_ci(a: np.ndarray, b: np.ndarray, seed: int) -> Tuple[float,float,float]:
    rng=np.random.default_rng(seed); delta=(b-a); n=len(delta)
    vals=np.empty(10000)
    for q in range(10000): vals[q]=np.mean(delta[rng.integers(0,n,n)])
    return float(np.mean(delta)),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

def make_pair_summary(a: pd.DataFrame,b: pd.DataFrame,label_a:str,label_b:str,seed:int) -> pd.DataFrame:
    a=a.set_index("scenario_id").sort_index(); b=b.set_index("scenario_id").sort_index(); rows=[]
    metrics=[c for c in a.columns if c not in ("random_seed","scenario_label","controller")]
    for metric in metrics:
        if metric in ("NotRecovered_count",): continue
        x=a[metric].to_numpy(float); y=b[metric].to_numpy(float)
        if np.all(np.isnan(x)) or np.all(np.isnan(y)): continue
        valid=~(np.isnan(x)|np.isnan(y)); x=x[valid];y=y[valid]
        delta,lo,hi=bootstrap_ci(x,y,seed+len(rows))
        rel=(np.mean(y)-np.mean(x))/np.mean(x)*100 if abs(np.mean(x))>1e-12 else np.nan
        dz=delta/np.std(y-x,ddof=1) if len(x)>1 and np.std(y-x,ddof=1)>1e-12 else np.nan
        rows.append({"comparison":f"{label_a} vs {label_b}","metric":metric,"n":len(x),"A_mean":float(np.mean(x)),"A_SD":float(np.std(x,ddof=1)),"B_mean":float(np.mean(y)),"B_SD":float(np.std(y,ddof=1)),"paired_relative_change_pct":rel,"paired_delta":delta,"bootstrap95_low":lo,"bootstrap95_high":hi,"paired_dz":dz})
    return pd.DataFrame(rows)

def robustness_summary(groups: Dict[str, List[Scenario]], pidpars: Dict[str,float]) -> pd.DataFrame:
    """Evaluate fixed severity levels using the same controller and metrics.

    The numerical severity is encoded in the scenario generator, so this is a
    sensitivity analysis rather than any claim about a plant uncertainty set.
    """
    rows=[]
    for label, scenarios in groups.items():
        for scn in scenarios:
            f,_,_=run_one(scn,"DMC-DV hold",pidpars,"hold")
            m=scenario_metrics(f,scn)
            for cv in ("HP_pH","MP_pH","COND_kappa"):
                rows.append({"severity":label,"scenario_id":scn.scenario_id,"CV":cv,
                             "IAE":m[f"IAE_{cv}"],"RMSE":m[f"RMSE_{cv}"],
                             "PeakDev":m[f"PeakDev_{cv}"],"TV_total":m[f"TV_{MV_NAMES[0]}"]+m[f"TV_{MV_NAMES[1]}"],
                             "DMC_active_ratio":m["DMC_active_ratio"],"QP_infeasible_count":m["QP_infeasible_count"]})
    per=pd.DataFrame(rows)
    summ=per.groupby(["severity","CV"],as_index=False).agg(IAE_mean=("IAE","mean"),IAE_SD=("IAE","std"),RMSE_mean=("RMSE","mean"),PeakDev_mean=("PeakDev","mean"),TV_total_mean=("TV_total","mean"),DMC_active_ratio_mean=("DMC_active_ratio","mean"),QP_infeasible_total=("QP_infeasible_count","sum"))
    return per,summ

def make_scenario_parameter_rows(scenarios: List[Scenario]) -> pd.DataFrame:
    rows=[]
    for s in scenarios:
        for iy,cv in enumerate(CV_NAMES):
            for j,name in enumerate(MV_NAMES+DV_NAMES):
                rows.append({"scenario_id":s.scenario_id,"random_seed":s.seed,"CV":cv,"input":name,"K_m":KM[iy,j],"K_p":s.Kp[iy,j],"delta_K":s.Kp[iy,j]/KM[iy,j]-1 if KM[iy,j]!=0 else np.nan,"theta_m_min":THETAM[iy,j],"theta_p_min":s.thetap[iy,j],"delta_theta_min":s.thetap[iy,j]-THETAM[iy,j],"tau_m_min":TAUM[iy,j],"tau_p_min":s.taup[iy,j],"delta_tau":s.taup[iy,j]/TAUM[iy,j]-1})
            rows.append({"scenario_id":s.scenario_id,"random_seed":s.seed,"CV":cv,"input":"initial_colored_disturbance","K_m":np.nan,"K_p":float(s.w[0,iy]),"delta_K":np.nan,"theta_m_min":np.nan,"theta_p_min":np.nan,"delta_theta_min":np.nan,"tau_m_min":np.nan,"tau_p_min":np.nan,"delta_tau":"saved initial state"})
        for ev in s.events:
            rows.append({"scenario_id":s.scenario_id,"random_seed":s.seed,"CV":"event","input":ev["dv"],"K_m":np.nan,"K_p":np.nan,"delta_K":np.nan,"theta_m_min":ev["start_min"],"theta_p_min":ev["duration_min"],"delta_theta_min":ev["dv_index"],"tau_m_min":ev["amplitude"],"tau_p_min":np.nan,"delta_tau":ev["kind"]})
    return pd.DataFrame(rows)

def plot_figures(pred_records, timeseries, pid_metrics, dmc_metrics, hold_metrics, trend_metrics, long_df, repsid):
    plt.rcParams.update({"font.size":8,"font.family":"Arial","axes.spines.top":False,"axes.spines.right":False})
    pred=pd.DataFrame(pred_records); rep=pred[(pred.scenario_id==repsid)&(pred.condition=="mismatch_noise")&(pred.horizon_step==10)]
    fig,ax=plt.subplots(2,3,figsize=(7.2,3.8),sharex='col')
    for i,cv in enumerate(CV_NAMES):
        g=rep[rep.CV==cv]
        ax[0,i].plot(g.time_min,g.measured_future,color="#333333",lw=1,label="Measured")
        ax[0,i].plot(g.time_min,g.predicted,color=COLORS["DMC-DV hold"],lw=1,label="Predicted")
        ax[0,i].set_title(cv.replace("_"," ")); ax[0,i].set_ylabel("pH" if i<2 else "μS/cm")
        ax[1,i].plot(g.time_min,g.predicted-g.measured_future,color="#666666",lw=.8); ax[1,i].axhline(0,color="#222222",lw=.5);ax[1,i].set_ylabel("Residual");ax[1,i].set_xlabel("Time (min)")
    # Use one figure-level legend above the panels so no trace is obscured.
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT/"Fig6_model_prediction_validation.png",dpi=350);plt.close(fig)
    pm=pid_metrics; dm=dmc_metrics
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.5))
    for axx,metric,title in zip(axes,["IAE_HP_pH","IAE_MP_pH","IAE_COND_kappa"],["HP pH IAE","MP pH IAE","Conductivity IAE"]):
        axx.boxplot([pm[metric],dm[metric]],labels=["PID","DMC trend"],patch_artist=True,boxprops={"facecolor":"#DDDDDD"},medianprops={"color":"#111111"})
        axx.collections[-1].set_facecolor(COLORS["Proposed DMC-DV trend"]) if axx.collections else None
        axx.set_title(title);axx.set_ylabel("Integrated absolute error")
    fig.tight_layout();fig.savefig(OUT/"Fig7_pid_vs_dmc_monte_carlo.png",dpi=350);plt.close(fig)
    rep=timeseries[timeseries.scenario_id==repsid]
    pid=rep[rep.controller=="PID"]; dmc=rep[rep.controller=="DMC-DV hold"]
    fig,axes=plt.subplots(3,1,figsize=(7.2,5.2),sharex=True)
    for dv in ("Cond_Flow","Makeup_Flow"):
        axes[0].plot(dmc.time_min,dmc[f"{dv}_true"],lw=1,label=dv.replace("_"," "))
    axes[0].set_ylabel("DV deviation");axes[0].legend(frameon=False,ncol=2,fontsize=7)
    for cv in CV_NAMES:
        axes[1].plot(pid.time_min,pid[f"{cv}_true"],color=COLORS["PID"],alpha=.45,lw=.8)
        axes[1].plot(dmc.time_min,dmc[f"{cv}_true"],color=COLORS["Proposed DMC-DV trend"],alpha=.75,lw=.8)
    axes[1].axhline(9.60,color="#999999",lw=.4);axes[1].axhline(9.70,color="#999999",lw=.4);axes[1].set_ylabel("CV (mixed units)")
    for mv in MV_NAMES:
        axes[2].plot(pid.time_min,pid[mv],color=COLORS["PID"],alpha=.6,lw=.8)
        axes[2].plot(dmc.time_min,dmc[mv],color=COLORS["Proposed DMC-DV trend"],alpha=.8,lw=.8)
    axes[2].set_ylabel("MV deviation");axes[2].set_xlabel("Time (min)");fig.tight_layout();fig.savefig(OUT/"Fig8_representative_disturbance.png",dpi=350);plt.close(fig)
    hold=rep[rep.controller=="DMC-DV hold"]; trend=rep[rep.controller=="Proposed DMC-DV trend"]
    fig,axes=plt.subplots(3,1,figsize=(7.2,5.2),sharex=True)
    for dv in ("Cond_Flow","Makeup_Flow"):
        axes[0].plot(trend.time_min,trend[f"{dv}_true"],color="#333333",lw=.8,label=f"{dv} true")
        axes[0].plot(trend.time_min,trend[f"{dv}_measured"],color="#999999",lw=.6,alpha=.8,label=f"{dv} measured")
    axes[0].legend(frameon=False,ncol=2,fontsize=6);axes[0].set_ylabel("DV deviation")
    for cv in CV_NAMES:
        axes[1].plot(hold.time_min,hold[f"{cv}_true"],color=COLORS["DMC-DV hold"],lw=.8,alpha=.7)
        axes[1].plot(trend.time_min,trend[f"{cv}_true"],color=COLORS["Proposed DMC-DV trend"],lw=.8,alpha=.7)
    axes[1].set_ylabel("CV (mixed units)")
    for mv in MV_NAMES:
        axes[2].plot(hold.time_min,hold[mv],color=COLORS["DMC-DV hold"],lw=.8)
        axes[2].plot(trend.time_min,trend[mv],color=COLORS["Proposed DMC-DV trend"],lw=.8)
    axes[2].set_ylabel("MV deviation");axes[2].set_xlabel("Time (min)");fig.tight_layout();fig.savefig(OUT/"Fig9_dv_prediction_ablation.png",dpi=350);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(7.2,5.2),sharex=True)
    day=long_df.time_min/1440
    axes[0].plot(day,long_df.HP_pH_true,color="#0072B2",lw=.5,label="HP pH");axes[0].plot(day,long_df.MP_pH_true,color="#009E73",lw=.5,label="MP pH");axes[0].axhspan(9.60,9.70,color="#EEEEEE",zorder=0);axes[0].set_ylabel("pH");axes[0].legend(frameon=False,ncol=2,fontsize=7)
    axes[1].plot(day,long_df.COND_kappa_true,color="#D55E00",lw=.5);axes[1].axhspan(0,.30,color="#EEEEEE",zorder=0);axes[1].set_ylabel("Conductivity (μS/cm)")
    axes[2].step(day,long_df.DMC_active,where="post",color="#0072B2",lw=.6,label="DMC active");axes[2].step(day,long_df.constraint_amplitude_active+long_df.constraint_rate_active+long_df.constraint_cv_active,where="post",color="#555555",lw=.5,label="Any active constraint");axes[2].set_ylim(-.05,1.2);axes[2].set_ylabel("State");axes[2].set_xlabel("Simulation day");axes[2].legend(frameon=False,ncol=2,fontsize=7)
    fig.tight_layout();fig.savefig(OUT/"Fig10_15d_long_term_simulation.png",dpi=350);plt.close(fig)

def main():
    pidpars=tune_pid()
    selected_p, horizon_sweep = select_prediction_horizon(pidpars)
    scenarios=[create_scenario(i, MASTER_SEED+i, 360, True, True, "reference: ±15% K, ±20% τ, ±2 min θ") for i in range(1,31)]
    nominal=[create_scenario(i, MASTER_SEED+i, 360, False, False, "nominal") for i in range(1,31)]
    # A deterministic negative test: the third CV is deliberately shifted
    # above its 0.30 zone bound so the one-sided conductivity penalty and its
    # hard upper bound are exercised in reported robustness results.
    cond_stress=[]
    for i in range(1,31):
        sc=create_scenario(100+i, MASTER_SEED+500+i, 360, True, True, "conductivity-zone stress")
        sc.w[:,2] += 0.175
        cond_stress.append(sc)
    ts_parts=[]; pred=[]; metrics=[]
    for scn,nom in zip(scenarios,nominal):
        for name,mode in [("PID",None),("DMC-DV hold","hold"),("Proposed DMC-DV trend","trend")]:
            f,pr,_=run_one(scn,name,pidpars,mode);ts_parts.append(f);metrics.append(scenario_metrics(f,scn))
            if name=="DMC-DV hold": pred.extend(pr)
        _,pr_nom,_=run_one(nom,"DMC-DV hold",pidpars,"hold"); pred.extend(pr_nom)
    stress_metrics=[]
    for scn in cond_stress:
        f,_,_=run_one(scn,"DMC-DV hold",pidpars,"hold")
        stress_metrics.append(scenario_metrics(f,scn))
    ts=pd.concat(ts_parts,ignore_index=True); metric=pd.DataFrame(metrics)
    pred_per,pred_summary=summarize_prediction(pred)
    pidm=metric[metric.controller=="PID"].copy(); trendm=metric[metric.controller=="Proposed DMC-DV trend"].copy(); holdm=metric[metric.controller=="DMC-DV hold"].copy()
    pid_dmc=make_pair_summary(pidm,holdm,"PID","DMC-DV hold",MASTER_SEED+500)
    ablation=make_pair_summary(holdm,trendm,"DMC-DV hold","Proposed DMC-DV trend",DV_TREND_BOOTSTRAP_SEED)
    # A separate 15 d, 1 min-resolution, robust offline simulation.
    long_scn=create_scenario(100, MASTER_SEED+1000, 21600, True, True, "15 d reference")
    # A few rare, short unmeasured chemistry excursions keep the 15 d robust
    # simulation informative without changing the specified AR(1) background
    # disturbance setting.  They emulate exceptional unmodelled water-quality
    # changes and are recorded as scenario events for full reproducibility.
    long_rng=np.random.default_rng(MASTER_SEED+2100)
    for idx, st in enumerate(range(1800, 21000, 2400)):
        dur=int(long_rng.integers(6, 13))
        amp=long_rng.choice([-1.0,1.0], size=NY)*np.array([0.045,0.045,0.060])
        long_scn.w[st:st+dur] += amp
        long_scn.events.append({"dv":"unmeasured_chemistry_excursion", "dv_index":-1,
                                "start_min":int(st), "amplitude":float(np.linalg.norm(amp)),
                                "kind":"short_unmeasured_excursion", "duration_min":dur})
    long_df,_,long_meta=run_one(long_scn,"DMC-DV hold",pidpars,"hold")
    long_metrics=scenario_metrics(long_df,long_scn)
    long_summary=pd.DataFrame([{**long_metrics,"duration_d":15,"simulation_type":"15 d long-horizon offline theoretical simulation"}])
    params={"simulation_type":"offline theoretical constrained-DMC benchmark","master_seed":MASTER_SEED,"Ts_min":TS,"N":N,"P":P,"M":M,"horizon_candidates":[10,20,30,40],"horizon_selection":"12 independent 360 min validation scenarios; feasibility first, then HP/IP IAE, then TV tie-break within 1%","pH_tracking_Q":"diag(4,4)","conductivity_zone_weight":KAPPA_ZONE_WEIGHT,"conductivity_zone_threshold":KAPPA_ZONE,"conductivity_predictive_hard_upper_constraint":KAPPA_HARD,"R_u":"diag(1.0,1.2)","u_bounds":"[-0.40,0.40]","du_rate":"±0.03/min","plant_gain_mismatch":"Uniform(-0.15,0.15)","plant_tau_mismatch":"Uniform(-0.20,0.20)","plant_theta_offset_min":"choice{-2,-1,0,1,2}","CV_measurement_noise_std":{"HP_pH":.002,"IP_pH":.002,"COND_kappa":.003},"DV_measurement_noise_std":DV_NOISE.tolist(),"colored_disturbance_ar":AR,"colored_disturbance_target_std":W_STD.tolist(),"PID_parameters":pidpars,"PID_tuning":"one held-out 240 min scenario, seed=MASTER_SEED+77; grid Kp1={3,5,7,9}, Ki1={0.04,0.08,0.12}, Kp2={3,5,7,9,11}, Ki2={0.04,0.08,0.12}; minimize IAE+75 saturation+0.015 TV","PID_feedback":"decentralized baseline: MV1→HP pH and MV2→IP pH; conductivity is a monitored constrained CV","DV_hold":"dhat(k+p|k)=d(k)","DV_trend":"L=5 bounded slope, no future true DV","QP":"standard slack-variable constrained QP solved by active set with Phase-I scipy linprog; hold then PID fallback after three consecutive infeasible periods","DMC3_boundary":"Aspen DMC3 is not used to generate numerical results; it is discussed only as an implementation mapping."}
    pd.DataFrame([{"parameter":k,"value":json.dumps(v) if isinstance(v,(dict,list)) else v} for k,v in params.items()]).to_csv(OUT/"simulation_parameters.csv",index=False)
    horizon_sweep.to_csv(OUT/"prediction_horizon_selection.csv",index=False)
    make_scenario_parameter_rows(scenarios).to_csv(OUT/"monte_carlo_scenario_parameters.csv",index=False)
    ts.to_csv(OUT/"monte_carlo_timeseries.csv",index=False)
    pd.concat([pred_per.assign(level="per_scenario"),pred_summary.assign(level="summary")],ignore_index=True,sort=False).to_csv(OUT/"model_prediction_metrics.csv",index=False)
    pid_dmc.to_csv(OUT/"pid_vs_dmc_metrics.csv",index=False)
    ablation.to_csv(OUT/"dv_ablation_metrics.csv",index=False)
    pd.DataFrame(stress_metrics).to_csv(OUT/"conductivity_zone_stress_metrics.csv",index=False)
    long_df.to_csv(OUT/"long_term_15d_timeseries.csv",index=False)
    long_summary.to_csv(OUT/"long_term_15d_summary.csv",index=False)
    (OUT/"scenario_events.json").write_text(json.dumps([{"scenario_id":s.scenario_id,"random_seed":s.seed,"events":s.events} for s in scenarios]+[long_meta],ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"simulation_manifest.json").write_text(json.dumps(params,ensure_ascii=False,indent=2),encoding="utf-8")
    # Fixed sensitivity set: only the plant uncertainty/noise severity changes.
    robust_groups={}
    for label,kg,kt,th,noise in [("low",.05,.10,1,.5),("reference",.15,.20,2,1.0),("high",.25,.35,3,1.5)]:
        group=[]
        for sid in range(1,21):
            sc=create_scenario(200+sid, MASTER_SEED+3000+sid, 360, True, True, label)
            rng=np.random.default_rng(MASTER_SEED+4000+sid)
            sc.Kp=KM*(1+rng.uniform(-kg,kg,KM.shape))
            sc.taup=TAUM*(1+rng.uniform(-kt,kt,TAUM.shape))
            sc.thetap=np.maximum(0,THETAM+rng.choice(np.arange(-th,th+1),THETAM.shape))
            sc.y_noise*=noise; sc.dv_meas_noise*=noise; sc.w*=noise
            group.append(sc)
        robust_groups[label]=group
    rob_per,rob_sum=robustness_summary(robust_groups,pidpars)
    rob_per.to_csv(OUT/"robustness_sensitivity_per_scenario.csv",index=False)
    rob_sum.to_csv(OUT/"robustness_sensitivity_summary.csv",index=False)
    plot_figures(pred,ts,pidm,holdm,holdm,trendm,long_df,7)
    print(json.dumps({"output_dir":str(OUT),"pid_parameters":pidpars,"pid_vs_dmc_rows":len(pid_dmc),"long_summary":long_summary.to_dict(orient="records")[0]},ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
