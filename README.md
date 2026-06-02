# ModalAI UAV System Identification

This repository contains tools for analyzing RCbenchmark dynamometer data for the ModalAI Starling 2 and Starling 2 Max UAV's (including Max and Custom Rotor configurations). 

The tools process raw step-sweep and step-response CSV logs to extract mathematical aerodynamic constants. These outputs map PWM signals to thrust, torque, current, and motor speed, giving you the parameters needed for accurate physics simulations (like Isaac Sim and Pegasus).

## Included Scripts

*   **`analyze_sweep.py`**: Extracts static performance limits and polynomial relationships (e.g., Thrust coefficient $k_T$, Torque coefficient $k_Q$, and overall efficiency) from a slow PWM ramp (sweep) test.
*   **`analyze_step_response.py`**: Evaluates the dynamic performance of the system by computing the motor time constant ($\tau$) from a step-response test.

## Usage

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
2. Run the analysis:
   ```bash
   uv run bash commands.sh
   ```

Outputs (plots, `sim_model.json`, `pegasus_motor_params.yaml`) are saved to `analysis/`.
