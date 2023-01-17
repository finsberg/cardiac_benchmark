import subprocess as sp

args = ["--alpha-m=0.0 --alpha-f=0.0", ""]

for arg in args:
    for i in range(1, 17):
        sp.run(["sbatch", "step2.sbatch", str(i), arg])
