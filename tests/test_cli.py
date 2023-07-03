import shutil
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from cardiac_benchmark.cli import app

here = Path(__file__).parent.absolute()


@pytest.fixture(scope="module")
def geo_path():
    runner = CliRunner()
    path = Path("test_geometry.h5")
    path.unlink(missing_ok=True)
    # Use a much coarser geometry to speed up the tests
    result = runner.invoke(
        app,
        ["create-geometry", path.as_posix(), "--mesh-size-factor=5.0"],
    )
    assert result.exit_code == 0
    yield path.absolute().as_posix()
    path.unlink(missing_ok=False)


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

    assert np.isclose(up0.max(), max_up0)
    assert np.isclose(time_stamps[-1], min_up0)

    shutil.rmtree(outdir)


def test_benchmark1_step0_caseA(geo_path):
    _benchmark_1(
        "benchmark1_step0_caseA",
        ["benchmark1-step0-case-a"],
        geo_path,
        max_up0=0.001214217740157648,
        min_up0=-0.02118657320131717,
    )


def test_benchmark1_step0_caseB(geo_path):
    _benchmark_1(
        "benchmark1_step0_caseB",
        ["benchmark1-step0-case-b"],
        geo_path,
        max_up0=0.009563210660943304,
        min_up0=-0.0002485061442104368,
    )


def test_benchmark1_step1(geo_path):
    _benchmark_1(
        "benchmark1_step1",
        ["benchmark1-step1"],
        geo_path,
        max_up0=0.0015262332937695028,
        min_up0=-0.020129923655389646,
    )


def test_benchmark1_step2_case1(geo_path):
    _benchmark_1(
        "benchmark1_step2",
        ["benchmark1-step2", "1"],
        geo_path,
        max_up0=0.0015262332937695028,
        min_up0=-0.020129923655389646,
    )
