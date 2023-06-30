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


def _benchmark_1(fname: str, cmd: str, _geo_path: str) -> None:
    reference_data = np.load(here / "data" / f"{fname}.npy", allow_pickle=True).item()

    # breakpoint()
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

    up1 = np.load(outdir / "componentwise_displacement_up1.npy")
    up0 = np.load(outdir / "componentwise_displacement_up0.npy")
    volume = np.load(outdir / "volume.npy")
    time_stamps = np.load(outdir / "time_stamps.npy")
    sp1 = np.load(outdir / "von_Mises_stress_sp1.npy")
    sp0 = np.load(outdir / "von_Mises_stress_sp0.npy")

    for key, data in [
        ("up0", up0),
        ("up1", up1),
        ("volume", volume),
        ("time_stamps", time_stamps),
        ("sp0", sp0),
        ("sp1", sp1),
    ]:
        ref_data = reference_data[key]
        assert np.max(np.subtract(ref_data, data)) / np.max(np.abs(ref_data)) < 1e-6

    shutil.rmtree(outdir)


def test_benchmark1_step0_caseA(geo_path):
    _benchmark_1("benchmark1_step0_caseA", ["benchmark1-step0-case-a"], geo_path)


def test_benchmark1_step0_caseB(geo_path):
    _benchmark_1("benchmark1_step0_caseB", ["benchmark1-step0-case-b"], geo_path)


def test_benchmark1_step1(geo_path):
    _benchmark_1("benchmark1_step1", ["benchmark1-step1"], geo_path)


def test_benchmark1_step2_case1(geo_path):
    _benchmark_1("benchmark1_step2", ["benchmark1-step2", "1"], geo_path)
