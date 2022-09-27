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


def main():
    for case in cases:
        benchmark.run_benchmark(**case)
        loader = DataLoader(case["outpath"])
        loader.postprocess_all()


if __name__ == "__main__":
    main()
