#starling 2 Max
python analyze_sweep.py data/starling2_max/starling2max_sweep_2026-05-26_201318_prev.csv --out analysis/starling2_max --mass-kg 0.5
python analyze_step_response.py data/starling2_max/starling2_max_stepresponse.csv --out analysis/starling2_max --yaml-path analysis/starling2_max/pegasus_motor_params.yaml

#starling 2 default rotor
python analyze_sweep.py data/starling2_defaultrotor/starling2_sweep_2026-05-27_212140.csv --out analysis/starling2_defaultrotor --mass-kg 0.3
python analyze_step_response.py data/starling2_defaultrotor/starling2_stepresponse.csv --out analysis/starling2_defaultrotor --yaml-path analysis/starling2_defaultrotor/pegasus_motor_params.yaml

#starling 2 custom motor
python analyze_sweep.py data/starling2_customrotor/starling2_sweep_2026-05-27_214436.csv --out analysis/starling2_customrotor --mass-kg 0.3
python analyze_step_response.py data/starling2_customrotor/starling2_stepresponse.csv --out analysis/starling2_customrotor --yaml-path analysis/starling2_customrotor/pegasus_motor_params.yaml