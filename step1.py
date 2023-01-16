import argparse
import datetime
import json
import pprint
from pathlib import Path
from typing import Optional
from typing import Sequence

import benchmark
from postprocess import DataLoader

cases = [
    {
        "alpha_epi": 1e6,
        "eta": 1e1,
        "a_f": 2e4,
        "sigma_0": 1e5,
        "outpath": "results/case1.h5",
    },
    {
        "alpha_epi": 1e6,
        "eta": 1e1,
        "a_f": 2e4,
        "sigma_0": 1e6,
        "outpath": "results/case2.h5",
    },
    {
        "alpha_epi": 1e6,
        "eta": 1e1,
        "a_f": 5e4,
        "sigma_0": 1e5,
        "outpath": "results/case3.h5",
    },
    {
        "alpha_epi": 1e6,
        "eta": 1e1,
        "a_f": 5e4,
        "sigma_0": 1e6,
        "outpath": "results/case4.h5",
    },
    {
        "alpha_epi": 1e6,
        "eta": 1e2,
        "a_f": 2e4,
        "sigma_0": 1e5,
        "outpath": "results/case5.h5",
    },
    {
        "alpha_epi": 1e6,
        "eta": 1e2,
        "a_f": 2e4,
        "sigma_0": 1e6,
        "outpath": "results/case6.h5",
    },
    {
        "alpha_epi": 1e6,
        "eta": 1e2,
        "a_f": 5e4,
        "sigma_0": 1e5,
        "outpath": "results/case7.h5",
    },
    {
        "alpha_epi": 1e6,
        "eta": 1e2,
        "a_f": 5e4,
        "sigma_0": 1e6,
        "outpath": "results/case8.h5",
    },
    {
        "alpha_epi": 1e8,
        "eta": 1e1,
        "a_f": 2e4,
        "sigma_0": 1e5,
        "outpath": "results/case9.h5",
    },
    {
        "alpha_epi": 1e8,
        "eta": 1e1,
        "a_f": 2e4,
        "sigma_0": 1e6,
        "outpath": "results/case10.h5",
    },
    {
        "alpha_epi": 1e8,
        "eta": 1e1,
        "a_f": 5e4,
        "sigma_0": 1e5,
        "outpath": "results/case11.h5",
    },
    {
        "alpha_epi": 1e8,
        "eta": 1e1,
        "a_f": 5e4,
        "sigma_0": 1e6,
        "outpath": "results/case12.h5",
    },
    {
        "alpha_epi": 1e8,
        "eta": 1e2,
        "a_f": 2e4,
        "sigma_0": 1e5,
        "outpath": "results/case13.h5",
    },
    {
        "alpha_epi": 1e8,
        "eta": 1e2,
        "a_f": 2e4,
        "sigma_0": 1e6,
        "outpath": "results/case14.h5",
    },
    {
        "alpha_epi": 1e8,
        "eta": 1e2,
        "a_f": 5e4,
        "sigma_0": 1e5,
        "outpath": "results/case15.h5",
    },
    {
        "alpha_epi": 1e8,
        "eta": 1e2,
        "a_f": 5e4,
        "sigma_0": 1e6,
        "outpath": "results/case16.h5",
    },
]


def main(argv: Optional[Sequence] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=int, help="The case number")
    parser.add_argument("-o", "--output", type=str, default="", help="Output directory")
    parser.add_argument("--no-run", action="store_true", help="Do not run benchmark")
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Do not run post-processing script",
    )
    args = vars(parser.parse_args(argv))
    case_nr = args["case"]
    case = cases[case_nr]
    run_benchmark = not args["no_run"]
    run_postprocess = not args["no_postprocess"]

    if args["output"] != "":
        outdir = Path(args["output"]).absolute()
    else:
        outdir = Path(__file__).absolute().parent

    case["outpath"] = outdir / "result.h5"

    parameters = case.copy()
    parameters["outpath"] = str(case["outpath"])
    parameters["timestamp"] = datetime.datetime.now().isoformat()
    parameters["case_nr"] = case_nr
    print(f"Running case {case_nr} with parameters:\n {pprint.pformat(case)}")
    print(f"Output will be saved to {outdir}")
    (outdir / "parameters.json").write_text(json.dumps(parameters))

    if run_benchmark:
        benchmark.run_benchmark(**case)

    if run_postprocess:
        loader = DataLoader(case["outpath"])
        loader.postprocess_all()
    return 0


if __name__ == "__main__":
    main()
