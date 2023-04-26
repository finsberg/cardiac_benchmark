import subprocess as sp
from pathlib import Path

here = Path(__file__).parent.absolute()
(Path.cwd() / "slurm-output").mkdir(exist_ok=True)
# args = ["--alpha-m=0.0 --alpha-f=0.0", ""]
args = ["--pressure=none"]

for arg in args:
    sp.run(["sbatch", (here / "step1.sbatch").as_posix(), arg])
