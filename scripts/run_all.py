"""Run the whole pipeline, Stage 1 to Stage 5, from the project folder:
    .venv\\Scripts\\python -m scripts.run_all
Each stage runs in its own process, so no state leaks between stages. Stops at the first failure.
"""
import subprocess
import sys

STAGES = [
    "stage1_size_design",
    "stage1_baseline",
    "stage1_validate",
    "stage2_attacks",
    "stage3_dataset",
    "stage4_detect",      # reads the Stage 3 dataset
    "stage5_resilience",  # reads the Stage 4 detection latencies
]


def main():
    for name in STAGES:
        print(f"\n===== {name} =====", flush=True)
        subprocess.run([sys.executable, "-m", f"scripts.{name}"], check=True)
    print("\nAll stages finished; results are in results/stage1 ... results/stage5.")


if __name__ == "__main__":
    main()
