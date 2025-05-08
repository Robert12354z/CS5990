import subprocess

print("Step 1: Preprocessing data...")
subprocess.run(["python", "data_load.py"], check=True)

print("\nStep 2: Training models...")
subprocess.run(["python", "train_models.py"], check=True)

print("\nStep 3: Testing models...")
subprocess.run(["python", "test_models.py"], check=True)

print("\nAll steps completed successfully.")
