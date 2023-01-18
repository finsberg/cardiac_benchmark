from pathlib import Path

import cardiac_benchmark

here = Path(__file__).absolute().parent


outdir = Path("/global/D1/homes/henriknf/cardiac_benchmark/step1")
disp_path = here / ".." / "data" / "displacement_points.npz"
vol_path = here / ".." / "data" / "computed_vols.npz"
for d in outdir.iterdir():
    loader = cardiac_benchmark.postprocess.DataLoader(d / "result.h5", outdir=d)
    loader.compare_results(folder=d, disp_path=disp_path, vol_path=vol_path)
