"""Console script for cardiac-benchmark."""
import datetime
import json
import pprint
from pathlib import Path
from typing import Optional

import typer

from . import benchmark
from . import step1 as _step1
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
    case: int,
    outdir: Optional[Path] = typer.Option(None),
    run_benchmark: bool = True,
    run_postprocess: bool = True,
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
    params = _step1.cases[case]

    parameters = {
        "case": case,
        "outdir": outdir.as_posix(),
        "outpath": outpath.as_posix(),
        "geometry_path": geometry_path,
        "timestamp": datetime.datetime.now().isoformat(),
        "parameters": params,
    }

    typer.echo(f"Running case {case} with parameters:\n {pprint.pformat(params)}")
    typer.echo(f"Output will be saved to {outdir}")
    (outdir / "parameters.json").write_text(json.dumps(parameters))

    if run_benchmark:
        benchmark.run(**params, outpath=outpath, geometry_path=geometry_path)

    if run_postprocess:
        loader = DataLoader(outpath)
        loader.postprocess_all(folder=outdir)

    return 0
