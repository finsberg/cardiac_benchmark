"""Console script for cardiac-benchmark."""
import datetime
import json
import pprint
from pathlib import Path
from typing import Optional

import typer

from . import benchmark
from . import step2 as _step2
from .postprocess import DataLoader

app = typer.Typer()


def version_callback(show_version: bool):
    """Prints version information."""
    if show_version:
        from . import __version__, __program_name__

        typer.echo(f"{__program_name__} {__version__}")
        raise typer.Exit()


def license_callback(show_license: bool):
    """Prints license information."""
    if show_license:
        from . import __license__

        typer.echo(f"{__license__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version",
    ),
    license: bool = typer.Option(
        None,
        "--license",
        callback=license_callback,
        is_eager=True,
        help="Show license",
    ),
):
    # Do other global stuff, handle other global options here
    return


@app.command(help="Run step 1")
def step1(
    outdir: Optional[Path] = typer.Option(None),
    run_benchmark: bool = True,
    run_postprocess: bool = True,
    run_comparison: bool = True,
    alpha_m: float = 0.2,
    alpha_f: float = 0.4,
    pressure: benchmark.Pressure = benchmark.Pressure.bestel,
    geometry_path: Optional[Path] = typer.Option(None),
) -> int:

    if outdir is not None:
        outdir = Path(outdir).absolute()
    else:
        outdir = Path.cwd() / "results"

    if geometry_path is None:
        geometry_path = Path.cwd() / "geometry.h5"

    outdir.mkdir(exist_ok=True, parents=True)
    outpath = outdir / "result.h5"

    params = benchmark.default_parameters()
    params["alpha_m"] = alpha_m
    params["alpha_f"] = alpha_f
    params["pressure"] = pressure
    params["outpath"] = outpath.as_posix()
    params["geometry_path"] = geometry_path.as_posix()

    parameters = params.copy()
    parameters["step"] = 1
    parameters["outdir"] = outdir.as_posix()
    parameters["timestamp"] = datetime.datetime.now().isoformat()

    typer.echo(f"Running step 1 with parameters {pprint.pformat(parameters)}")
    typer.echo(f"Output will be saved to {outdir}")
    (outdir / "parameters.json").write_text(json.dumps(parameters))

    if run_benchmark:
        benchmark.run(**params)  # type: ignore

    loader = DataLoader(outpath)
    if run_postprocess:
        loader.postprocess_all(folder=outdir)

    if run_comparison:
        loader.compare_results(folder=outdir)

    return 0


@app.command(help="Run step 2")
def step2(
    case: int,
    outdir: Optional[Path] = typer.Option(None),
    run_benchmark: bool = True,
    run_postprocess: bool = True,
    alpha_m: float = 0.2,
    alpha_f: float = 0.4,
    pressure: benchmark.Pressure = benchmark.Pressure.bestel,
    geometry_path: Optional[Path] = typer.Option(None),
) -> int:

    if outdir is not None:
        outdir = Path(outdir).absolute()
    else:
        outdir = Path.cwd() / "results"

    if geometry_path is None:
        geometry_path = Path.cwd() / "geometry.h5"

    outdir.mkdir(exist_ok=True, parents=True)
    outpath = outdir / "result.h5"

    params = benchmark.default_parameters()
    assert 1 <= case <= 16, "Case must be a number between 1 and 16"
    params.update(_step2.cases[case - 1])
    params["outpath"] = outpath.as_posix()
    params["geometry_path"] = geometry_path.as_posix()
    params["t_sys"] = 0.005
    params["t_dias"] = 0.319
    params["alpha_m"] = alpha_m
    params["alpha_f"] = alpha_f
    params["pressure"] = pressure

    parameters = params.copy()
    parameters["step"] = 2
    parameters["case"] = case
    parameters["outdir"] = outdir.as_posix()
    parameters["timestamp"] = datetime.datetime.now().isoformat()

    typer.echo(
        f"Running step 2, case {case} with parameters:\n {pprint.pformat(params)}",
    )
    typer.echo(f"Output will be saved to {outdir}")
    (outdir / "parameters.json").write_text(json.dumps(parameters))

    if run_benchmark:
        benchmark.run(**params)  # type: ignore

    if run_postprocess:
        loader = DataLoader(outpath)
        loader.postprocess_all(folder=outdir)

    return 0
