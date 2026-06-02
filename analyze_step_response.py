"""analyze_step_response.py
Identify a first-order time constant from RCbenchmark step-response CSV.

Model:    dT/dt = (T_setpoint - T) / tau
          T(t) = T_final + (T_initial - T_final) * exp(-(t - t_step) / tau)

Approach: locate transitions using the `pwm_commanded_us` column logged by
03_step_response.js, fit a first-order exponential to each transition,
report mean tau across transitions.

Usage:
    python analyze_step_response.py path/to/stepresp.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for n in candidates:
        if n in df.columns:
            return n
    raise KeyError(f"None of {candidates} in {list(df.columns)}")


def first_order(t, T_final, T_initial, tau, t0):
    return T_final + (T_initial - T_final) * np.exp(-np.maximum(t - t0, 0.0) / tau)


def detect_outliers(fits: list[dict], mode: str) -> tuple[list[bool], dict]:
    """Return a boolean mask (True = inlier) plus a diagnostics dict.

    Combines two filters:
      - IQR x 1.5 on tau (Tukey fences) — catches statistical outliers
      - rms_error_N relative to step magnitude (|T_final - T_initial|) >
        rms_frac — catches bad fits regardless of tau value

    `mode` selects behavior:
      - "both":  apply both filters
      - "iqr":   IQR only
      - "rms":   RMS-quality only
      - "off":   no filtering (everyone is an inlier)
    """
    rms_frac = 0.03  # 3% of step magnitude — tuned to catch fits where the
                     # initial-condition guess landed mid-transient (high RMS)
    taus = np.array([f["tau_s"] for f in fits])
    rms  = np.array([f["rms_error_N"] for f in fits])
    step_mag = np.array([abs(f["T_final"] - f["T_initial"]) for f in fits])

    if mode == "off" or len(fits) < 4:
        return [True] * len(fits), {"mode": mode, "n_dropped": 0}

    q1, q3 = np.percentile(taus, [25, 75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    iqr_keep = (taus >= lo) & (taus <= hi)

    rms_threshold = rms_frac * np.maximum(step_mag, 1.0)
    rms_keep = rms <= rms_threshold

    if mode == "iqr":
        keep = iqr_keep
    elif mode == "rms":
        keep = rms_keep
    else:  # "both"
        keep = iqr_keep & rms_keep

    return keep.tolist(), {
        "mode": mode,
        "iqr_fences_ms": [float(lo) * 1000, float(hi) * 1000],
        "rms_frac": rms_frac,
        "n_dropped": int((~keep).sum()),
        "dropped_indices": [i for i, k in enumerate(keep) if not k],
    }


def patch_yaml_tau(yaml_path: Path, tau_s: float, source_csv: str,
                   n_used: int, n_total: int, std_ms: float) -> bool:
    """Replace the placeholder motor_time_constant line in pegasus_motor_params.yaml.

    Looks for either `motor_time_constant: null` or `motor_time_constant: <num>`
    and replaces with the new tau plus a provenance comment block. Returns True
    on success.
    """
    if not yaml_path.exists():
        return False
    lines = yaml_path.read_text().splitlines()

    # Find the motor_time_constant: line
    target_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("motor_time_constant:"):
            target_idx = i
            break
    if target_idx is None:
        return False

    indent_line = lines[target_idx]
    stripped = indent_line.lstrip()
    indent = indent_line[: len(indent_line) - len(stripped)]

    # Walk backward to find the start of the immediately-preceding comment
    # block (so re-runs don't accumulate stale comments).
    block_start = target_idx
    while block_start > 0:
        prev = lines[block_start - 1].lstrip()
        if prev.startswith("#"):
            block_start -= 1
        else:
            break

    new_block = [
        f"{indent}# First-order motor time constant (seconds).",
        f"{indent}# Source: {source_csv}",
        f"{indent}# Patched by analyze_step_response.py",
        f"{indent}# Robust median across {n_used}/{n_total} fits "
        f"(std {std_ms:.0f} ms across inliers)",
        f"{indent}motor_time_constant: {tau_s:.4f}",
    ]

    out = lines[:block_start] + new_block + lines[target_idx + 1:]
    yaml_path.write_text("\n".join(out) + "\n")
    return True


def fit_one_transition(t: np.ndarray, T: np.ndarray) -> dict:
    T_initial_guess = float(T[0])
    T_final_guess   = float(T[-1])
    tau_guess       = max(0.02, (t[-1] - t[0]) / 5.0)
    t0_guess        = float(t[0])
    p0 = [T_final_guess, T_initial_guess, tau_guess, t0_guess]
    try:
        popt, pcov = curve_fit(first_order, t, T, p0=p0, maxfev=5000)
        T_final, T_initial, tau, t0 = popt
        residual = T - first_order(t, *popt)
        rms = float(np.sqrt(np.mean(residual ** 2)))
        return {
            "tau_s":     float(tau),
            "T_initial": float(T_initial),
            "T_final":   float(T_final),
            "t0":        float(t0),
            "rms_error_N": rms,
            "n_samples": int(len(t)),
            "ok": True,
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e), "n_samples": int(len(t))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=Path("analysis_out"))
    ap.add_argument("--time-col", default="Time (s)",
                    help="Time column header in CSV")
    ap.add_argument("--window-s", type=float, default=3.0,
                    help="Window size after each step edge to fit (s). "
                         "Should cover ~3-5 tau plus settling; default tuned "
                         "for ~200-500 ms tau and dwellTime_s=6.")
    ap.add_argument("--outlier-filter", choices=["both", "iqr", "rms", "off"],
                    default="both",
                    help="Outlier-rejection mode for aggregated tau. "
                         "iqr=Tukey fences on tau; rms=fit-quality threshold; "
                         "both=AND of the two; off=no filtering.")
    ap.add_argument("--yaml-path", type=Path,
                    default=Path("analysis_out/pegasus_motor_params.yaml"),
                    help="Pegasus YAML to patch with the identified tau.")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

    t_col = find_col(df, [args.time_col, "Time (s)", "time"])
    T_col = find_col(df, ["Thrust (kgf)", "Thrust (gf)", "Thrust (g)", "Thrust"])
    pwm_col = find_col(df, ["pwm_commanded_us", "ESC signal (µs)", "ESC signal (us)", "ESC signal"])

    t   = df[t_col].to_numpy(dtype=float)
    T_raw = df[T_col].to_numpy(dtype=float)
    # Convert thrust to Newtons based on the unit in the header.
    T_N = T_raw * (9.80665 if "kgf" in T_col else 9.80665e-3)
    pwm = df[pwm_col].to_numpy(dtype=float)

    # Detect step edges (PWM changes)
    edges = np.where(np.diff(pwm) != 0)[0] + 1
    if len(edges) == 0:
        print("No step edges detected.", file=sys.stderr)
        return 2
    print(f"Found {len(edges)} step edges at indices: {edges[:10]}{'...' if len(edges) > 10 else ''}")

    fits = []
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, T_N, "k-", lw=0.7, alpha=0.6, label="thrust (N)")

    for k, idx in enumerate(edges):
        t_step = t[idx]
        mask = (t >= t_step) & (t <= t_step + args.window_s)
        if mask.sum() < 5:
            continue
        ts = t[mask]
        Ts = T_N[mask]
        f = fit_one_transition(ts, Ts)
        if not f["ok"]:
            continue
        fits.append({
            **f,
            "step_idx": int(k),
            "pwm_after": float(pwm[idx]),
        })
        # Overlay fit
        t_dense = np.linspace(ts[0], ts[-1], 100)
        ax.plot(t_dense, first_order(t_dense, f["T_final"], f["T_initial"], f["tau_s"], f["t0"]),
                "r-", lw=1.5, alpha=0.8)
        ax.axvline(t_step, color="gray", ls=":", alpha=0.4)

    ax.set(xlabel="Time (s)", ylabel="Thrust (N)",
           title=f"Step response — {args.csv.name}")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p_png = args.out / "step_response.png"
    fig.savefig(p_png, dpi=120)
    print(f"  wrote {p_png}")

    # Outlier detection runs before the overlay plot so the model curve
    # uses the robust (inlier-only) tau, not the raw median.
    if fits:
        early_keep, _ = detect_outliers(fits, args.outlier_filter)
        early_taus_in = np.array([f["tau_s"] for f, k in zip(fits, early_keep) if k])
        tau_for_overlay = float(np.median(early_taus_in)) if len(early_taus_in) else float(np.median([f["tau_s"] for f in fits]))
    else:
        tau_for_overlay = None

    # ---- Overlay plot: step-up & step-down cycles superimposed -----------
    # Each cycle is shown centered on its step command (t_rel = 0). Window
    # is t_step - 2 s to t_step + 5 s. Step-up on top, step-down on bottom,
    # shared x and y limits so the two are directly comparable. The fitted
    # first-order model (using median tau across inliers) is overlaid in
    # bold black so the data scatter around it is visible.
    t_pre, t_post = -2.0, 5.0
    fig2, (ax_up, ax_dn) = plt.subplots(2, 1, figsize=(11, 9), sharex=True, sharey=True)
    cmap = plt.get_cmap("tab10")
    pwm_diff = np.diff(pwm)
    n_up = n_dn = 0
    up_T_lo, up_T_hi, dn_T_lo, dn_T_hi = [], [], [], []

    def plot_cycle(ax, t_rel, T_rel, cycle_idx, color):
        """Plot one cycle: connect with a line, mark fit-window points big,
        and non-fit points small. The fit window is [0, args.window_s]."""
        in_fit  = (t_rel >= 0) & (t_rel <= args.window_s)
        out_fit = ~in_fit
        # Single connecting line through all points (so the line is continuous)
        ax.plot(t_rel, T_rel, "-", lw=1.0, color=color, alpha=0.55, zorder=2,
                label=f"cycle {cycle_idx + 1}")
        # Larger markers for points that were used in curve_fit
        ax.plot(t_rel[in_fit], T_rel[in_fit], "o", ms=6.5, color=color,
                alpha=0.95, zorder=4, mec="black", mew=0.4)
        # Smaller markers for points that were NOT used (pre-step or tail)
        ax.plot(t_rel[out_fit], T_rel[out_fit], "o", ms=3.0, color=color,
                alpha=0.5, zorder=3)

    for k, idx in enumerate(edges):
        t_step = t[idx]
        mask = (t >= t_step + t_pre) & (t <= t_step + t_post)
        if mask.sum() < 2:
            continue
        t_rel = t[mask] - t_step
        T_rel = T_N[mask]
        direction_up = pwm_diff[idx - 1] > 0
        # Capture pre-step and post-settled values for the model curve
        pre_mask = t_rel < 0
        post_mask = t_rel > 1.5  # ~5+ tau after step, should be settled
        if direction_up:
            if pre_mask.any():
                up_T_lo.append(np.mean(T_rel[pre_mask]))
            if post_mask.any():
                up_T_hi.append(np.mean(T_rel[post_mask]))
            plot_cycle(ax_up, t_rel, T_rel, n_up, cmap(n_up % 10))
            n_up += 1
        else:
            if pre_mask.any():
                dn_T_hi.append(np.mean(T_rel[pre_mask]))
            if post_mask.any():
                dn_T_lo.append(np.mean(T_rel[post_mask]))
            plot_cycle(ax_dn, t_rel, T_rel, n_dn, cmap(n_dn % 10))
            n_dn += 1

    # Model curve uses robust tau (median of inliers, computed above).
    if tau_for_overlay is not None:
        tau_model = tau_for_overlay
        T_low_model = float(np.mean(up_T_lo)) if up_T_lo else 1.0
        T_high_model = float(np.mean(up_T_hi)) if up_T_hi else 18.0
        # Step-up curve
        t_curve = np.linspace(t_pre, t_post, 400)
        T_up_curve = np.where(
            t_curve < 0,
            T_low_model,
            T_high_model + (T_low_model - T_high_model) * np.exp(-t_curve / tau_model),
        )
        T_dn_curve = np.where(
            t_curve < 0,
            T_high_model,
            T_low_model + (T_high_model - T_low_model) * np.exp(-t_curve / tau_model),
        )
        ax_up.plot(t_curve, T_up_curve, "k-", lw=3.2, alpha=0.9, zorder=10,
                   label=f"model (τ={tau_model*1000:.0f} ms)")
        ax_dn.plot(t_curve, T_dn_curve, "k-", lw=3.2, alpha=0.9, zorder=10,
                   label=f"model (τ={tau_model*1000:.0f} ms)")

    # Mark the step instant
    ax_up.axvline(0.0, color="0.4", ls=":", lw=1)
    ax_dn.axvline(0.0, color="0.4", ls=":", lw=1)

    ax_up.set(ylabel="Thrust (N)",
              title=f"Step UP (low → hover) — {n_up} cycles")
    ax_up.grid(alpha=0.3); ax_up.legend(loc="lower right", fontsize=8, ncol=2)
    ax_dn.set(xlabel="Time since step (s)", ylabel="Thrust (N)",
              title=f"Step DOWN (hover → low) — {n_dn} cycles")
    ax_dn.grid(alpha=0.3); ax_dn.legend(loc="upper right", fontsize=8, ncol=2)
    ax_up.set_xlim(t_pre, t_post)
    # y limits sized for the step magnitude (with a small margin)
    fig2.suptitle(args.csv.stem, fontsize=10, y=1.00)
    fig2.tight_layout()
    p_overlay = args.out / "step_response_overlay.png"
    fig2.savefig(p_overlay, dpi=120, bbox_inches="tight")
    print(f"  wrote {p_overlay}")

    if fits:
        taus = np.array([f["tau_s"] for f in fits])
        keep_mask, diag = detect_outliers(fits, args.outlier_filter)
        for i, f in enumerate(fits):
            f["outlier"] = not keep_mask[i]
        taus_in = taus[np.array(keep_mask)]
        if len(taus_in) == 0:
            print("All fits flagged as outliers — keeping raw values.", file=sys.stderr)
            taus_in = taus
            keep_mask = [True] * len(fits)
        summary = {
            "tau_s_robust_median": float(np.median(taus_in)),
            "tau_s_robust_mean":   float(np.mean(taus_in)),
            "tau_s_robust_std":    float(np.std(taus_in)),
            "tau_s_raw_mean":      float(np.mean(taus)),
            "tau_s_raw_median":    float(np.median(taus)),
            "tau_s_raw_std":       float(np.std(taus)),
            "n_transitions":       len(fits),
            "n_inliers":           int(sum(keep_mask)),
            "outlier_filter":      diag,
            "per_transition":      fits,
        }
        p_json = args.out / "step_response_fit.json"
        p_json.write_text(json.dumps(summary, indent=2))
        print(f"  wrote {p_json}")
        print(f"\nIdentified motor time constant tau (mode: {args.outlier_filter}):")
        print(f"  Raw      : mean = {summary['tau_s_raw_mean']*1000:6.1f} ms, "
              f"median = {summary['tau_s_raw_median']*1000:6.1f} ms, "
              f"std = {summary['tau_s_raw_std']*1000:5.1f} ms, n = {len(fits)}")
        if diag["n_dropped"] > 0:
            dropped_taus = [f"{taus[i]*1000:.0f} ms" for i in diag["dropped_indices"]]
            print(f"  Dropped  : {diag['n_dropped']} outlier(s) — {dropped_taus}")
        else:
            print(f"  Dropped  : 0 outliers (filter mode: {args.outlier_filter})")
        print(f"  Robust   : mean = {summary['tau_s_robust_mean']*1000:6.1f} ms, "
              f"median = {summary['tau_s_robust_median']*1000:6.1f} ms, "
              f"std = {summary['tau_s_robust_std']*1000:5.1f} ms, n = {summary['n_inliers']}")
        if summary["tau_s_robust_mean"] < 0.200:
            print("  WARNING: tau < 200 ms is at the scripting-engine log-rate")
            print("  limit (~2-3 Hz, file-I/O bound). Treat as an upper bound;")
            print("  consider longer dwell times or more cycles.")

        # Patch the Pegasus YAML in-place with the robust median tau
        ok = patch_yaml_tau(
            args.yaml_path,
            tau_s=summary["tau_s_robust_median"],
            source_csv=args.csv.name,
            n_used=summary["n_inliers"],
            n_total=len(fits),
            std_ms=summary["tau_s_robust_std"] * 1000,
        )
        if ok:
            print(f"  patched motor_time_constant in {args.yaml_path}")
        else:
            print(f"  NOTE: could not patch {args.yaml_path} (file missing or no "
                  f"motor_time_constant key found)")
    else:
        print("No successful fits.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
