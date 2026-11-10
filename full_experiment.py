"""Resume full training, then evaluate LFW/QMUL and write the final report."""
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "work" / "full" / "pipeline_status.json"


def status(value):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    value["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(STATUS)


def main():
    stages = [
        ("training", ["run_project.py", "train", "--log-file", "work/full_training.log"]),
        ("lfw_evaluation", ["run_project.py", "recognition", "--benchmark", "lfw", "--log-file", "work/full_lfw.log"]),
        ("qmul_evaluation", ["run_project.py", "recognition", "--benchmark", "qmul", "--log-file", "work/full_qmul.log"]),
        ("report", ["write_full_report.py"]),
    ]
    completed = []
    for name, arguments in stages:
        status({"state": "running", "stage": name, "completed_stages": completed})
        print("FULL EXPERIMENT STAGE", name, flush=True)
        start = time.perf_counter()
        result = subprocess.run([sys.executable, "-u", *arguments], cwd=ROOT)
        if result.returncode:
            status({"state": "failed", "stage": name, "returncode": result.returncode, "completed_stages": completed})
            raise SystemExit(result.returncode)
        completed.append({"stage": name, "seconds_this_invocation": time.perf_counter() - start})
    status({"state": "complete", "completed_stages": completed})
    print("FULL EXPERIMENT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
