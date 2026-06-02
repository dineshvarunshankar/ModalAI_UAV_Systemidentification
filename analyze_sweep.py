"""analyze_sweep.py
Parse an RCbenchmark step-sweep CSV and produce:
    analysis_out/sweep_plots.png   -- thrust/RPM/power/efficiency vs PWM
    analysis_out/sweep_fits.png    -- fitted polynomial overlays
    analysis_out/sim_model.json    -- coefficients for RL sim

Usage:
    python analyze_sweep.py path/to/log.csv [--mass-kg 2.5]

Tested column-name fallbacks cover RCbenchmark v1.x and v2.x exports.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# RCbenchmark CSV column names vary between firmware versions and unit
# preferences. We accept several aliases per logical channel.
ALIASES = {
    "pwm":       ["ESC signal (µs)", "ESC signal (us)", "ESC signal", "esc"],
    "thrust":    ["Thrust (kgf)", "Thrust (gf)", "Thrust (g)", "Thrust"],
    "torque_Nm": ["Torque (N·m)", "Torque (N⋅m)", "Torque (N.m)", "Torque (Nm)", "Torque"],
    "rpm":       ["Motor Electrical Speed (RPM)",
                  "Motor Optical Speed (RPM)",
                  "Motor Speed (RPM)"],
    "current_A": ["Current (A)"],
    "voltage_V": ["Voltage (V)"],
}


def thrust_to_N(values: np.ndarray, col_name: str) -> np.ndarray:
    """Convert thrust column to Newtons based on the unit in the header."""
    if "kgf" in col_name:
        return values * 9.80665
    # grams (gf, g, or unitless legacy "Thrust" column assumed grams)
    return values * 9.80665e-3


def find_col(df: pd.DataFrame, key: str) -> str:
    for name in ALIASES[key]:
        if name in df.columns:
            return name
    raise KeyError(f"No column for {key!r}; columns are {list(df.columns)}")


def fit_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares fit y = k * x with no intercept."""
    x = x.reshape(-1, 1)
    return float(np.linalg.lstsq(x, y, rcond=None)[0][0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=Path("analysis_out"))
    ap.add_argument("--mass-kg", type=float, default=2.5,
                    help="Per-motor share of vehicle mass (for hover marker)")
    ap.add_argument("--poly-order", type=int, default=3)
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

    pwm       = df[find_col(df, "pwm")].to_numpy(dtype=float)
    thrust_col = find_col(df, "thrust")
    thrust_N  = thrust_to_N(df[thrust_col].to_numpy(dtype=float), thrust_col)
    torque    = df[find_col(df, "torque_Nm")].to_numpy(dtype=float)
    rpm       = df[find_col(df, "rpm")].to_numpy(dtype=float)
    current   = df[find_col(df, "current_A")].to_numpy(dtype=float)
    voltage   = df[find_col(df, "voltage_V")].to_numpy(dtype=float)

    # Filter out invalid / pre-arm rows
    valid = (pwm >= 1050) & (rpm > 100) & np.isfinite(thrust_N)
    pwm, thrust_N, torque = pwm[valid], thrust_N[valid], torque[valid]
    rpm, current, voltage = rpm[valid], current[valid], voltage[valid]

    if len(pwm) == 0:
        print("No valid rows after filtering.", file=sys.stderr)
        return 2

    # Sort by PWM in case the sweep wasn't monotonic
    order = np.argsort(pwm)
    pwm, thrust_N, torque = pwm[order], thrust_N[order], torque[order]
    rpm, current, voltage = rpm[order], current[order], voltage[order]

    omega = rpm * 2.0 * np.pi / 60.0     # rad/s
    P_elec = voltage * current           # W
    # Use the magnitude of torque for the propeller model fit. Sign on the
    # bench depends on load-cell wiring vs prop direction; the sim model
    # wants |kQ| and assigns yaw sign per-rotor in the mixer.
    torque_abs = np.abs(torque)
    P_mech = torque_abs * omega          # W
    eta_drive = np.where(P_elec > 1.0, P_mech / P_elec, np.nan)
    eff_NperW = np.where(P_elec > 1.0, thrust_N / P_elec, np.nan)

    # Static prop coefficients: T = kT * omega^2 ; |tau| = kQ * omega^2
    kT = fit_through_origin(omega ** 2, thrust_N)
    kQ = fit_through_origin(omega ** 2, torque_abs)
    torque_sign = "positive" if np.median(torque) > 0 else "negative"
    print(f"  Bench-measured torque sign: {torque_sign} (kQ reported as |kQ|)")

    # Throttle command in [0, 1]
    u = (pwm - 1000.0) / 1000.0

    p_T = np.polyfit(u, thrust_N, args.poly_order)         # highest-order first
    p_w = np.polyfit(u, omega,    args.poly_order)
    p_I = np.polyfit(u, current,  args.poly_order)
    p_Q = np.polyfit(u, torque_abs, args.poly_order)

    # ---- Plot 1: raw curves ---------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes[0, 0].plot(pwm, thrust_N, "o-")
    axes[0, 0].axhline(args.mass_kg * 9.80665, ls="--", alpha=0.4,
                       label=f"hover ({args.mass_kg} kg/motor)")
    axes[0, 0].set(xlabel="PWM (us)", ylabel="Thrust (N)"); axes[0, 0].grid(); axes[0, 0].legend()

    axes[0, 1].plot(pwm, rpm, "o-")
    axes[0, 1].set(xlabel="PWM (us)", ylabel="RPM"); axes[0, 1].grid()

    axes[1, 0].plot(pwm, P_elec, "o-", label="P electrical")
    axes[1, 0].plot(pwm, P_mech, "s--", label="P mechanical")
    axes[1, 0].set(xlabel="PWM (us)", ylabel="Power (W)"); axes[1, 0].grid(); axes[1, 0].legend()

    axes[1, 1].plot(pwm, eff_NperW, "o-", color="tab:green")
    axes[1, 1].set(xlabel="PWM (us)", ylabel="Thrust efficiency (N/W)")
    axes[1, 1].grid()

    fig.suptitle(args.csv.name)
    fig.tight_layout()
    p_raw = args.out / "sweep_plots.png"
    fig.savefig(p_raw, dpi=120)
    print(f"  wrote {p_raw}")

    # ---- Plot 2: fits ----------------------------------------------------
    u_dense = np.linspace(u.min(), u.max(), 200)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(u, thrust_N, "o", label="data")
    axes[0].plot(u_dense, np.polyval(p_T, u_dense), "-", label=f"poly ord {args.poly_order}")
    axes[0].set(xlabel="u (normalized throttle)", ylabel="Thrust (N)"); axes[0].grid(); axes[0].legend()
    omega_dense = np.linspace(omega.min(), omega.max(), 200)

    axes[1].plot(omega ** 2, thrust_N, "o", label="data")
    axes[1].plot(omega_dense ** 2, kT * omega_dense ** 2, "-", label=f"T = kT*omega², kT={kT:.3e}")
    axes[1].set(xlabel="omega² (rad²/s²)", ylabel="Thrust (N)"); axes[1].grid(); axes[1].legend()

    fig.tight_layout()
    p_fits = args.out / "sweep_fits.png"
    fig.savefig(p_fits, dpi=120)
    print(f"  wrote {p_fits}")

    # ---- Plot 3: prop performance curves (KDE-style) --------------------
    # Two stacked panels: Thrust vs RPM (top), Torque vs RPM (bottom).
    # Fitted quadratic curve drawn behind the data points.
    rpm_dense = np.linspace(0.0, rpm.max() * 1.05, 300)
    omega_dense_perf = rpm_dense * 2.0 * np.pi / 60.0
    fig, axes = plt.subplots(2, 1, figsize=(9, 9))

    axes[0].plot(rpm_dense, kT * omega_dense_perf ** 2,
                 color="0.55", lw=2.5, zorder=1,
                 label=f"T = kT·ω², kT = {kT:.3e}")
    axes[0].scatter(rpm, thrust_N, s=42, color="tab:blue", edgecolor="black",
                    linewidth=0.7, zorder=3, label="measured")
    axes[0].set(xlabel="RPM", ylabel="Thrust [N]",
                title="Calculated Thrust")
    axes[0].grid(alpha=0.4); axes[0].legend(loc="upper left")
    axes[0].set_xlim(0, rpm.max() * 1.05)
    axes[0].set_ylim(bottom=0)

    axes[1].plot(rpm_dense, kQ * omega_dense_perf ** 2,
                 color="0.55", lw=2.5, zorder=1,
                 label=f"|τ| = kQ·ω², kQ = {kQ:.3e}")
    axes[1].scatter(rpm, torque_abs, s=42, color="tab:orange", edgecolor="black",
                    linewidth=0.7, zorder=3, label="measured")
    axes[1].set(xlabel="RPM", ylabel="Torque [Nm]",
                title="Calculated Torque")
    axes[1].grid(alpha=0.4); axes[1].legend(loc="upper left")
    axes[1].set_xlim(0, rpm.max() * 1.05)
    axes[1].set_ylim(bottom=0)

    fig.suptitle(args.csv.stem, fontsize=11, y=1.00)
    fig.tight_layout()
    p_perf = args.out / "prop_performance.png"
    fig.savefig(p_perf, dpi=120, bbox_inches="tight")
    print(f"  wrote {p_perf}")

    # ---- Sim model JSON --------------------------------------------------
    omega_max = float(omega.max())
    model = {
        "source_csv": str(args.csv),
        "n_points": int(len(pwm)),
        "voltage_nominal_V": float(np.median(voltage)),
        "static_prop_model": {
            "kT_N_s2_per_rad2": kT,
            "kQ_Nm_s2_per_rad2": kQ,
        },
        "polynomial_fits": {
            "throttle_to_thrust_N":   p_T.tolist(),
            "throttle_to_omega_rads": p_w.tolist(),
            "throttle_to_current_A":  p_I.tolist(),
            "throttle_to_torque_Nm":  p_Q.tolist(),
            "polynomial_order":       args.poly_order,
            "throttle_definition":    "u = (pwm_us - 1000) / 1000",
        },
        "summary": {
            "pwm_range_us":    [float(pwm.min()),    float(pwm.max())],
            "thrust_range_N":  [float(thrust_N.min()), float(thrust_N.max())],
            "rpm_max":         float(rpm.max()),
            "omega_max_rad_s": omega_max,
            "current_max_A":   float(current.max()),
            "P_elec_max_W":    float(P_elec.max()),
            "best_eff_NperW":  float(np.nanmax(eff_NperW)),
            "best_eff_pwm_us": float(pwm[np.nanargmax(eff_NperW)]) if np.isfinite(np.nanmax(eff_NperW)) else None,
        },
    }
    p_json = args.out / "sim_model.json"
    p_json.write_text(json.dumps(model, indent=2))
    print(f"  wrote {p_json}")

    # ---- Pegasus / Isaac Sim YAML stub -----------------------------------
    # tau_motor is filled in by analyze_step_response.py if you merge later.
    # We leave a placeholder here so the file is usable as soon as it lands.
    pegasus_yaml = f"""# Pegasus quadrotor motor parameters
# Source: {args.csv.name}
# Generated by analyze_sweep.py
#
# NOTE: field names below follow the Pegasus 'Multirotor' / 'Quadrotor'
# config convention. Verify against your AirStack/Pegasus fork — names
# may differ slightly across versions. The physical meaning is the same.

motor:
  # T = rotor_constant * omega^2  [N, omega in rad/s]
  rotor_constant: {kT:.6e}

  # tau_z = rolling_moment_coefficient * omega^2  [N*m]
  rolling_moment_coefficient: {kQ:.6e}

  # ratio: c_M / c_T  (used in some Pegasus versions instead of c_M)
  yaw_to_thrust_ratio: {(kQ / kT) if kT > 0 else float('nan'):.6e}

  # Maximum rotor angular velocity observed during sweep (rad/s)
  max_rotor_velocity: {omega_max:.3f}

  # First-order motor time constant (seconds).
  # PLACEHOLDER: run `python analyze_step_response.py <stepresp.csv>` next.
  # That script edits this file in-place and replaces the null below with
  # the identified tau (and prepends a source comment).
  motor_time_constant: null  # !! NEEDS PATCHING — DO NOT USE THIS VALUE !!

  # Throttle command interpretation
  throttle:
    pwm_min_us: {float(pwm.min()):.0f}
    pwm_max_us: {float(pwm.max()):.0f}
    # Throttle u = (pwm_us - 1000) / 1000, in [0, 1] over [1000, 2000] us
    polynomial_throttle_to_thrust_N:   {p_T.tolist()}
    polynomial_throttle_to_omega_rads: {p_w.tolist()}

power_envelope:
  voltage_nominal_V: {float(np.median(voltage)):.2f}
  current_max_observed_A: {float(current.max()):.2f}
  electrical_power_max_W: {float(P_elec.max()):.1f}
  thrust_max_observed_N: {float(thrust_N.max()):.2f}
"""
    p_yaml = args.out / "pegasus_motor_params.yaml"
    p_yaml.write_text(pegasus_yaml)
    print(f"  wrote {p_yaml}")

    print("\nKey numbers:")
    print(f"  kT  = {kT:.4e}  N·s²/rad²")
    print(f"  kQ  = {kQ:.4e}  N·m·s²/rad²")
    print(f"  Max thrust observed: {thrust_N.max():.2f} N "
          f"({thrust_N.max() / 9.80665:.2f} kgf)")
    print(f"  Max current observed: {current.max():.2f} A")
    print(f"  Best efficiency: {model['summary']['best_eff_NperW']:.3f} N/W "
          f"at PWM {model['summary']['best_eff_pwm_us']} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
