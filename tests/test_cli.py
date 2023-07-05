import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from cardiac_benchmark.cli import app

here = Path(__file__).parent.absolute()


@pytest.fixture(scope="module")
def geo_path():
    yield (here / "test_geometry.h5").absolute().as_posix()


def _benchmark_1(
    fname: str,
    cmd: str,
    _geo_path: str,
    max_up0: float,
    min_up0: float,
) -> None:
    runner = CliRunner(mix_stderr=False)
    outdir = Path(f"test_{fname}")
    if outdir.is_dir():
        shutil.rmtree(outdir)

    # Use P_1 to speed up the tests
    result = runner.invoke(
        app,
        [
            *cmd,
            "--geometry-path",
            _geo_path,
            "--outdir",
            outdir.as_posix(),
            "--function-space=P_1",
        ],
    )

    assert result.exit_code == 0

    up0 = np.load(outdir / "componentwise_displacement_up0.npy")
    time_stamps = np.load(outdir / "time_stamps.npy")

    assert np.isclose(time_stamps[0], 0.001)
    assert np.isclose(time_stamps[-1], 0.999)

    computed_max = up0.max(0)
    computed_min = up0.min(0)

    assert np.allclose(computed_max, max_up0)
    assert np.allclose(computed_min, min_up0)

    shutil.rmtree(outdir)


def test_benchmark1_step0_caseA(geo_path):
    _benchmark_1(
        "benchmark1_step0_caseA",
        ["benchmark1-step0-case-a"],
        geo_path,
        max_up0=(0.0, 0.00113854, 0.00121422),
        min_up0=(-2.11865732e-02, -7.07775436e-06, -3.90594965e-04),
    )


def test_benchmark1_step0_caseB(geo_path):
    _benchmark_1(
        "benchmark1_step0_caseB",
        ["benchmark1-step0-case-b"],
        geo_path,
        max_up0=(9.56321066e-03, 4.23419184e-05, 1.52382853e-04),
        min_up0=(6.58201731e-08, -2.48506144e-04, -1.50213958e-08),
    )


def test_benchmark1_step1(geo_path):
    _benchmark_1(
        "benchmark1_step1",
        ["benchmark1-step1"],
        geo_path,
        max_up0=(0.00029864, 0.00152623, 0.0012752),
        min_up0=(-2.01299237e-02, -4.59802750e-06, -3.08946192e-04),
    )


def test_benchmark1_step2_case1(geo_path):
    _benchmark_1(
        "benchmark1_step2",
        ["benchmark1-step2", "1"],
        geo_path,
        max_up0=(0.00029864, 0.00152623, 0.0012752),
        min_up0=(-2.01299237e-02, -4.59802750e-06, -3.08946192e-04),
    )


def test_benchmark2():
    runner = CliRunner(mix_stderr=False)
    data_folder = Path.cwd() / "coarse_data"
    res_download = runner.invoke(
        app,
        ["download-data-benchmark2", "coarse", "--outdir", data_folder.as_posix()],
    )
    assert res_download.exit_code == 0

    outdir = Path.cwd() / "test_results_benchmark2"
    # Let us mock out the actual solving to speed it up
    with patch("cardiac_benchmark.solver.NonlinearSolver.solve") as mock_solve:
        mock_solve.return_value = (0, True)
        result = runner.invoke(
            app,
            [
                "benchmark2",
                data_folder.as_posix(),
                "--outdir",
                outdir.as_posix(),
                "--t",
                0.05,
            ],
        )

    assert result.exit_code == 0

    assert (outdir / "result.h5").is_file()
    assert (outdir / "von_Mises_stress_sp1.npy").is_file()
