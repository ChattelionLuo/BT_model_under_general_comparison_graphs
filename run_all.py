import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FIGURES = tuple(f"Figure{i}.py" for i in range(1, 7))
PARALLEL_FIGURES = {"Figure1.py", "Figure2.py", "Figure6.py"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    for figure in FIGURES:
        command = [sys.executable, str(ROOT / figure)]
        if args.workers is not None and figure in PARALLEL_FIGURES:
            command.extend(["--workers", str(args.workers)])
        print(f"Generating {figure}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"Six PNG and PDF figures are in {ROOT / 'results'}")
