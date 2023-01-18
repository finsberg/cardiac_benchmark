import json
from pathlib import Path

import numpy as np

outdir = Path("/global/D1/homes/henriknf/cardiac_benchmark/step2")

newmark_data = {}
ga_data = {}

print(r"\begin{tabular}{l|c|c|c|c}")
print(
    r"case & $\mathrm{max}|u(p_0)|$ & $\mathrm{max}|u(p_1)|$ & $\mathrm{max}|\sigma(p_0)|$ & $\mathrm{max}|\sigma(p_1)|$ \\",
)
print(r"\hline")

for casedir in outdir.iterdir():
    case_nr = int(casedir.stem.lstrip("case"))

    for jobdir in casedir.iterdir():
        params = json.loads((jobdir / "parameters.json").read_text())
        up0 = np.load(jobdir / "componentwise_displacement_up0.npy")
        up1 = np.load(jobdir / "componentwise_displacement_up1.npy")

        sp0 = np.load(jobdir / "von_Mises_stress_sp0.npy")
        sp1 = np.load(jobdir / "von_Mises_stress_sp1.npy")
        data = {
            "max_up0": np.linalg.norm(up0, axis=1).max(),
            "max_up1": np.linalg.norm(up1, axis=1).max(),
            "max_sp0": np.abs(sp0).max(),
            "max_sp1": np.abs(sp1).max(),
        }
        if params["alpha_m"] == params["alpha_m"] == 0:
            newmark_data[case_nr] = data
        else:
            print(f"{case_nr} & ", " & ".join(f"{v:.3f}" for v in data.values()), r"\\")
            ga_data[case_nr] = data
print(r"\end{tabular}")
(outdir / "features.json").write_text(json.dumps(ga_data, indent=2))
# up0_max = []
