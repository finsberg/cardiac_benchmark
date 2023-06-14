import subprocess as sp
from pathlib import Path

here = Path(__file__).parent.absolute()
(Path.cwd() / "slurm-output").mkdir(exist_ok=True)

args = [
    "",
    "--pressure=none",
    "--function-space=P_2",
    "--function-space=P_2 --pressure=none",
]

for arg in args:
    sp.run(["sbatch", (here / "step1.sbatch").as_posix(), arg])
