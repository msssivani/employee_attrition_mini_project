import subprocess
import sys


def execute_pipeline():

    print("[INFO] =========================================")
    print("[INFO] Starting Lab 3: Employee Attrition Baseline ML Pipeline")
    print("[INFO] =========================================")

    scripts = [
        "src/preprocess.py",
        "src/train.py",
        "src/evaluate.py"
    ]

    for script in scripts:

        print(f"\n[INFO] ---> Executing {script}...")

        result = subprocess.run(
            [sys.executable, script]
        )

        if result.returncode != 0:

            print(
                f"[ERROR] Pipeline halted. "
                f"{script} failed with exit code "
                f"{result.returncode}."
            )

            sys.exit(1)

    print(
        "\n[SUCCESS] Employee Attrition Lab 3 "
        "Pipeline fully executed!"
    )


if __name__ == "__main__":
    execute_pipeline()