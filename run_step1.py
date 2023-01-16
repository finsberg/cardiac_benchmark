import subprocess as sp

for i in range(1, 17):
    sp.run(["sbatch", "step1.sbatch", str(i)])
