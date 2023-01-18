import subprocess as sp

args = ["--alpha-m=0.0 --alpha-f=0.0", ""]

for arg in args:
    sp.run(["sbatch", "step1.sbatch", arg])
