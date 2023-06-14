import json
import numpy as np
import shutil
from pathlib import Path

# import cardiac_benchmark
# import matplotlib.pyplot as plt

step2dir = Path("/global/D1/homes/henriknf/cardiac_benchmark/step2")


TABLE = r"""
\begin{tabular}{l|c|c|c|c}
case & $\mathrm{max}|u(p_0)|$ & $\mathrm{max}|u(p_1)|$ & $\mathrm{max}|\sigma(p_0)|$ & $\mathrm{max}|\sigma(p_1)|$ \\
\hline
"""


def gather_zero_pressure():
    features = {}
    outdir = step2dir / "step2_zero_pressure_results_P1"
    outdir.mkdir(exist_ok=True)
    table = TABLE

    for casedir in step2dir.iterdir():
        if "case" not in casedir.stem:
            continue
        case_nr = int(casedir.stem.lstrip("case"))

        for jobdir in casedir.iterdir():
            print(jobdir)
            parameter_file = jobdir / "parameters.json"
            if not parameter_file.is_file():
                continue

            params = json.loads(parameter_file.read_text())

            if params["alpha_m"] == params["alpha_m"] == 0:
                # This using newmark beta method but we are using the
                # generalized alpha method
                continue

            if params.get("pressure") != "none":
                continue

            if not (jobdir / "componentwise_displacement_up0.npy").is_file():
                continue

            print("Found results!")
            # loader = cardiac_benchmark.postprocess.DataLoader(jobdir / "result.h5")

            up0 = np.load(jobdir / "componentwise_displacement_up0.npy")
            up1 = np.load(jobdir / "componentwise_displacement_up1.npy")

            sp0 = np.load(jobdir / "von_Mises_stress_sp0.npy")
            sp1 = np.load(jobdir / "von_Mises_stress_sp1.npy")
            features[case_nr] = {
                "max_up0": np.linalg.norm(up0, axis=1).max(),
                "max_up1": np.linalg.norm(up1, axis=1).max(),
                "max_sp0": np.abs(sp0).max(),
                "max_sp1": np.abs(sp1).max(),
            }

            # cardiac_benchmark.postprocess.plot_componentwise_displacement(
            #     up0=up0,
            #     up1=up1,
            #     time_stamps=loader.time_stamps,
            #     fname=outdir / f"case_{case_nr}_componentwise_displacement.png",
            # )
            # cardiac_benchmark.postprocess.plot_von_Mises_stress(
            #     sp0=sp0,
            #     sp1=sp1,
            #     time_stamps=loader.time_stamps,
            #     fname=outdir / f"case_{case_nr}_von_Mises.png",
            # )
            # np.save(outdir / f"case_{case_nr}_time", loader.time_stamps)
            # np.save(outdir / f"case_{case_nr}_u_p0", up0)
            # np.save(outdir / f"case_{case_nr}_u_p1", up1)
            # np.save(outdir / f"case_{case_nr}_vonMises_p0", sp0)
            # np.save(outdir / f"case_{case_nr}_vonMises_p1", sp1)
            # (outdir / f"case_{case_nr}_features.json").write_text(
            #     json.dumps(features, indent=2)
            # )

            table += (
                f"{case_nr} & "
                + " & ".join(f"{v:.3f}" for v in features[case_nr].values())
                + r"\\"
                + "\n"
            )

    table += r"\end{tabular}"
    (outdir / "features_zero_pressure.text").write_text(table)
    (outdir / "all_features_zero_pressure.json").write_text(
        json.dumps(features, indent=2),
    )
    shutil.make_archive(base_name="step2", format="zip", root_dir=outdir)
    shutil.copy("step2.zip", step2dir / "step2_zero_pressure_P1.zip")


def gather():
    table = TABLE
    features = {}
    outdir = step2dir / "step2_results_P1"
    outdir.mkdir(exist_ok=True)

    for casedir in step2dir.iterdir():
        if "case" not in casedir.stem:
            continue
        case_nr = int(casedir.stem.lstrip("case"))

        for jobdir in casedir.iterdir():
            print(jobdir)
            parameter_file = jobdir / "parameters.json"
            if not parameter_file.is_file():
                continue

            params = json.loads(parameter_file.read_text())

            if params["alpha_m"] == params["alpha_m"] == 0:
                # This using newmark beta method but we are using the
                # generalized alpha method
                continue

            if params.get("pressure") != "bestel":
                continue

            if not (jobdir / "componentwise_displacement_up0.npy").is_file():
                continue

            print("Found results!")
            # loader = cardiac_benchmark.postprocess.DataLoader(jobdir / "result.h5")

            up0 = np.load(jobdir / "componentwise_displacement_up0.npy")
            up1 = np.load(jobdir / "componentwise_displacement_up1.npy")

            sp0 = np.load(jobdir / "von_Mises_stress_sp0.npy")
            sp1 = np.load(jobdir / "von_Mises_stress_sp1.npy")
            features[case_nr] = {
                "max_up0": np.linalg.norm(up0, axis=1).max(),
                "max_up1": np.linalg.norm(up1, axis=1).max(),
                "max_sp0": np.abs(sp0).max(),
                "max_sp1": np.abs(sp1).max(),
            }

            # cardiac_benchmark.postprocess.plot_componentwise_displacement(
            #     up0=up0,
            #     up1=up1,
            #     time_stamps=loader.time_stamps,
            #     fname=outdir / f"case_{case_nr}_componentwise_displacement.png",
            # )
            # cardiac_benchmark.postprocess.plot_von_Mises_stress(
            #     sp0=sp0,
            #     sp1=sp1,
            #     time_stamps=loader.time_stamps,
            #     fname=outdir / f"case_{case_nr}_von_Mises.png",
            # )
            # np.save(outdir / f"case_{case_nr}_time", loader.time_stamps)
            # np.save(outdir / f"case_{case_nr}_u_p0", up0)
            # np.save(outdir / f"case_{case_nr}_u_p1", up1)
            # np.save(outdir / f"case_{case_nr}_vonMises_p0", sp0)
            # np.save(outdir / f"case_{case_nr}_vonMises_p1", sp1)
            # (outdir / f"case_{case_nr}_features.json").write_text(
            #     json.dumps(features, indent=2)
            # )

            table += (
                f"{case_nr} & "
                + " & ".join(f"{v:.3f}" for v in features[case_nr].values())
                + r"\\"
                + "\n"
            )

    table += r"\end{tabular}"
    (outdir / "features.text").write_text(table)
    (outdir / "all_features.json").write_text(json.dumps(features, indent=2))
    shutil.make_archive(base_name="step2", format="zip", root_dir=outdir)
    shutil.copy("step2.zip", step2dir / "step2_P1.zip")


if __name__ == "__main__":
    gather()
    gather_zero_pressure()
