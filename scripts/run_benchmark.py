import subprocess as sp
import sys
from pathlib import Path

here = Path(__file__).parent.absolute()


def run_step0_caseA(dry_run: bool = False):
    print("Run step 0 case A")
    if not dry_run:
        sp.run(["sbatch", (here / "step0-casea.sbatch").as_posix()])


def run_step0_caseB(dry_run: bool = False):
    print("Run step 0 case B")
    if not dry_run:
        sp.run(["sbatch", (here / "step0-caseb.sbatch").as_posix()])


def run_step1(dry_run: bool = False):
    print("Run step 1")
    if not dry_run:
        sp.run(["sbatch", (here / "step1.sbatch").as_posix()])


def run_step2(dry_run: bool = False):
    print("Run step 2")
    for i in range(1, 17):
        print(f"Case {i}")
        if not dry_run:
            sp.run(["sbatch", (here / "step2.sbatch").as_posix(), str(i)])


def run_benchmark2_coarse(dry_run: bool = False):
    print("Run benchmark 2 (coarse)")
    if not dry_run:
        sp.run(["sbatch", (here / "benchmark2_coarse.sbatch").as_posix()])


def run_benchmark2_fine(dry_run: bool = False):
    print("Run benchmark 2 (fine)")
    if not dry_run:
        sp.run(["sbatch", (here / "benchmark2_fine.sbatch").as_posix()])


def main(args):

    (Path.cwd() / "slurm-output").mkdir(exist_ok=True)
    if "step0-case-a" in args or "all" in args:
        run_step0_caseA(dry_run="--dry-run" in args)
    if "step0-case-b" in args or "all" in args:
        run_step0_caseB(dry_run="--dry-run" in args)
    if "step1" in args or "all" in args:
        run_step1(dry_run="--dry-run" in args)
    if "step2" in args or "all" in args:
        run_step2(dry_run="--dry-run" in args)
    if "benchmark2-coarse" in args or "all" in args:
        run_benchmark2_coarse(dry_run="--dry-run" in args)
    if "benchmark2-fine" in args or "all" in args:
        run_benchmark2_fine(dry_run="--dry-run" in args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
