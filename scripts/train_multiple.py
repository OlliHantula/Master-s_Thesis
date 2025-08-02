import subprocess

# Define the 8 configurations
# CHECK THE TARGETS
configs = [
    {"sequence_length": 200, "stride": 200, "target": "DBP", "use_rppg": "false", "rppg_mode": None, "batch_size": 8, "lr": 1e-5, "run_id": "_200_DBP_100e"},
    {"sequence_length": 200, "stride": 200, "target": "DBP", "use_rppg": "true", "rppg_mode": "sequence", "batch_size": 8, "lr": 1e-5, "run_id": "_200_DBP_rPPG_100e"},
    {"sequence_length": 200, "stride": 200, "target": "SBP", "use_rppg": "false", "rppg_mode": None, "batch_size": 8, "lr": 1e-5, "run_id": "_200_SBP_100e"},
    {"sequence_length": 200, "stride": 200, "target": "SBP", "use_rppg": "true", "rppg_mode": "sequence", "batch_size": 8, "lr": 1e-5, "run_id": "_200_SBP_rPPG_100e"},
    {"sequence_length": 25, "stride": 25, "target": "SBP", "use_rppg": "false", "rppg_mode": None, "batch_size": 8, "lr": 1e-5, "run_id": "_25_SBP_100e"},
    {"sequence_length": 25, "stride": 25, "target": "SBP", "use_rppg": "true", "rppg_mode": "frame", "batch_size": 8, "lr": 1e-5, "run_id": "_25_SBP_rPPG_100e"},
    {"sequence_length": 25, "stride": 25, "target": "DBP", "use_rppg": "false", "rppg_mode": None, "batch_size": 8, "lr": 1e-5, "run_id": "_25_DBP_100e"},
    {"sequence_length": 25, "stride": 25, "target": "DBP", "use_rppg": "true", "rppg_mode": "frame", "batch_size": 8, "lr": 1e-5, "run_id": "_25_DBP_rPPG_100e"}
]

# Run train.py with each configuration
for i, cfg in enumerate(configs, 1):
    cmd = [
        "python", "scripts/train.py",
        f"--sequence_length={cfg['sequence_length']}",
        f"--stride={cfg['stride']}",
        f"--target={cfg['target']}",
        f"--use_rppg={cfg['use_rppg']}",
        f"--batch_size={cfg['batch_size']}",
        f"--epochs=100",
        f"--lr={cfg['lr']}",
        f"--run_id={cfg['run_id']}"
    ]
    if cfg["use_rppg"] and cfg["rppg_mode"]:
        cmd.append(f"--rppg_mode={cfg['rppg_mode']}")

    print(f"Running configuration {i}: {cmd}")
    subprocess.run(cmd)

