"""Console script for cardiac-benchmark."""
import datetime
import json
import pprint
from pathlib import Path
from typing import Optional
from typing import Union

import dolfin
import typer

from . import benchmark1
from . import step2 as _step2
from .geometry import EllipsoidGeometry
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


def benchmark1_step_0_1(
    step: int,
    case: Union[str, int] = 0,
    outdir: Optional[Path] = typer.Option(None),
    run_benchmark: bool = True,
    run_postprocess: bool = True,
    run_comparison: bool = True,
    alpha_m: float = 0.2,
    alpha_f: float = 0.4,
    pressure: benchmark1.Pressure = benchmark1.Pressure.bestel,
    geometry_path: Optional[Path] = typer.Option(None),
    function_space: str = "P_2",
) -> int:
    if outdir is not None:
        outdir = Path(outdir).absolute()
    else:
        outdir = Path.cwd() / "results"

    if geometry_path is None:
        geometry_path = Path.cwd() / "geometry.h5"

    outdir.mkdir(exist_ok=True, parents=True)
    outpath = outdir / "result.h5"

    params = benchmark1.default_parameters()
    if case == 2:
        assert isinstance(case, int), "Case must be an integer for step 2"
        assert 1 <= case <= 16, "Case must be a number between 1 and 16"
        params.update(_step2.cases[case - 1])
    params["alpha_m"] = alpha_m
    params["alpha_f"] = alpha_f
    params["pressure"] = pressure
    params["outpath"] = outpath.as_posix()
    params["geometry_path"] = geometry_path.as_posix()
    params["function_space"] = function_space

    parameters = params.copy()
    parameters["step"] = step
    parameters["case"] = case
    parameters["outdir"] = outdir.as_posix()
    parameters["timestamp"] = datetime.datetime.now().isoformat()

    typer.echo(
        f"Running step {step}, case {case} with parameters {pprint.pformat(parameters)}",
    )
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        typer.echo(f"Output will be saved to {outdir}")
        (outdir / "parameters.json").write_text(json.dumps(parameters))

    if run_benchmark:
        benchmark1.run(**params)  # type: ignore

    loader = DataLoader(outpath)
    if run_postprocess:
        loader.postprocess_all(folder=outdir)

    if run_comparison:
        loader.compare_results(folder=outdir)

    return 0


@app.command(help="Run benchmark 1 - step 0 - case A")
def benchmark1_step0_case_A(
    outdir: Optional[Path] = typer.Option(None),
    run_benchmark: bool = True,
    run_postprocess: bool = True,
    run_comparison: bool = True,
    alpha_m: float = 0.2,
    alpha_f: float = 0.4,
    geometry_path: Optional[Path] = typer.Option(None),
    function_space: str = "P_2",
) -> int:
    if outdir is None:
        outdir = Path("results_benchmark1_step0_caseA")
    return benchmark1_step_0_1(
        step=0,
        case="A",
        outdir=outdir,
        run_benchmark=run_benchmark,
        run_postprocess=run_postprocess,
        run_comparison=run_comparison,
        alpha_m=alpha_m,
        alpha_f=alpha_f,
        pressure=benchmark1.Pressure.zero_pressure,
        geometry_path=geometry_path,
        function_space=function_space,
    )


@app.command(help="Run benchmark 1 - step 0 - case B")
def benchmark1_step0_case_B(
    outdir: Optional[Path] = typer.Option(None),
    run_benchmark: bool = True,
    run_postprocess: bool = True,
    run_comparison: bool = True,
    alpha_m: float = 0.2,
    alpha_f: float = 0.4,
    geometry_path: Optional[Path] = typer.Option(None),
    function_space: str = "P_2",
) -> int:
    if outdir is None:
        outdir = Path("results_benchmark1_step0_caseB")
    return benchmark1_step_0_1(
        step=0,
        case="B",
        outdir=outdir,
        run_benchmark=run_benchmark,
        run_postprocess=run_postprocess,
        run_comparison=run_comparison,
        alpha_m=alpha_m,
        alpha_f=alpha_f,
        pressure=benchmark1.Pressure.zero_active,
        geometry_path=geometry_path,
        function_space=function_space,
    )


@app.command(help="Run benchmark 1 - step 1")
def benchmark1_step1(
    outdir: Optional[Path] = typer.Option(None),
    run_benchmark: bool = True,
    run_postprocess: bool = True,
    run_comparison: bool = True,
    alpha_m: float = 0.2,
    alpha_f: float = 0.4,
    geometry_path: Optional[Path] = typer.Option(None),
    function_space: str = "P_2",
) -> int:
    if outdir is None:
        outdir = Path("results_benchmark1_step1")
    return benchmark1_step_0_1(
        step=1,
        outdir=outdir,
        run_benchmark=run_benchmark,
        run_postprocess=run_postprocess,
        run_comparison=run_comparison,
        alpha_m=alpha_m,
        alpha_f=alpha_f,
        pressure=benchmark1.Pressure.bestel,
        geometry_path=geometry_path,
        function_space=function_space,
    )


@app.command(help="Run benchmark 1 - step 2")
def benchmark1_step2(
    case: int,
    outdir: Optional[Path] = typer.Option(None),
    run_benchmark: bool = True,
    run_postprocess: bool = True,
    alpha_m: float = 0.2,
    alpha_f: float = 0.4,
    geometry_path: Optional[Path] = typer.Option(None),
    function_space: str = "P_2",
) -> int:
    if outdir is None:
        outdir = Path(f"results_benchmark1_step2/case{case}")
    return benchmark1_step_0_1(
        step=2,
        case=case,
        outdir=outdir,
        run_benchmark=run_benchmark,
        run_postprocess=run_postprocess,
        run_comparison=False,
        alpha_m=alpha_m,
        alpha_f=alpha_f,
        pressure=benchmark1.Pressure.bestel,
        geometry_path=geometry_path,
        function_space=function_space,
    )


@app.command(help="Create geometry")
def create_geometry(
    path: Path,
    alpha_endo: float = -60.0,
    alpha_epi: float = 60.0,
    function_space: str = "Quadrature_4",
):
    geo = EllipsoidGeometry.from_parameters(
        fiber_params={
            "alpha_endo": alpha_endo,
            "alpha_epi": alpha_epi,
            "function_space": function_space,
        },
    )
    geo.save(path)
