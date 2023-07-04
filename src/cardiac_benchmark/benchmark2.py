import logging
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Union

import dolfin
import numpy as np

from . import activation_model
from . import postprocess
from . import pressure_model
from .geometry import BiVGeometry
from .material import HolzapfelOgden
from .problem import BiVProblem
from .utils import _update_parameters

logger = logging.getLogger(__name__)


def solve(
    problem: BiVProblem,
    tau: dolfin.Constant,
    activation: np.ndarray,
    lv_pressure: np.ndarray,
    rv_pressure: np.ndarray,
    plv: dolfin.Constant,
    prv: dolfin.Constant,
    time: np.ndarray,
    collector: postprocess.DataCollector,
    store_freq: int = 1,
) -> None:
    for i, (t, a, plv_, prv_) in enumerate(
        zip(time, activation, lv_pressure, rv_pressure),
    ):
        logger.info(
            f"{i}: Solving for time {t:.3f} with tau = {a:.3f}, lvp = {plv_:.3f} and rvp = {prv_:.3f}",
        )

        tau.assign(a)
        plv.assign(plv_)
        prv.assign(prv_)
        converged = problem.solve()
        if not converged:
            raise RuntimeError

        if i % store_freq == 0:
            dolfin.info("Store solution")
            collector.store(t)


def default_parameters():
    return dict(
        problem_parameters=BiVProblem.default_parameters(),
        material_parameters=HolzapfelOgden.default_parameters(),
        lv_pressure_parameters=pressure_model.default_lv_parameters_benchmark2(),
        rv_pressure_parameters=pressure_model.default_rv_parameters_benchmark2(),
        activation_parameters=activation_model.default_parameters(),
    )


def run(
    mesh_file: Path,
    fiber_file: Path,
    sheet_file: Path,
    sheet_normal_file: Path,
    activation_parameters: Optional[Dict[str, float]] = None,
    lv_pressure_parameters: Optional[Dict[str, float]] = None,
    rv_pressure_parameters: Optional[Dict[str, float]] = None,
    material_parameters: Optional[Dict[str, Union[float, dolfin.Constant]]] = None,
    problem_parameters: Optional[Dict[str, Union[float, dolfin.Constant]]] = None,
    outpath: Union[str, Path] = "results_benchmark2.h5",
):
    outdir = Path(outpath).parent
    outdir.mkdir(parents=True, exist_ok=True)

    geo = BiVGeometry.from_files(
        mesh_file=mesh_file,
        fiber_file=fiber_file,
        sheet_file=sheet_file,
        sheet_normal_file=sheet_normal_file,
    )

    problem_parameters = _update_parameters(
        BiVProblem.default_parameters(),
        problem_parameters,
    )
    material_parameters = _update_parameters(
        HolzapfelOgden.default_parameters(),
        material_parameters,
    )
    lv_pressure_parameters = _update_parameters(
        pressure_model.default_lv_parameters_benchmark2(),
        lv_pressure_parameters,
    )
    rv_pressure_parameters = _update_parameters(
        pressure_model.default_rv_parameters_benchmark2(),
        rv_pressure_parameters,
    )
    activation_parameters = _update_parameters(
        activation_model.default_parameters(),
        activation_parameters,
    )

    dt = float(problem_parameters["dt"])
    tau = dolfin.Constant(0.0)
    time = np.arange(dt, 1, dt)

    t_eval = time - float(problem_parameters["alpha_f"]) * dt
    lv_pressure = pressure_model.pressure_function(
        (0, 1),
        t_eval=t_eval,
        parameters=lv_pressure_parameters,
    )
    rv_pressure = pressure_model.pressure_function(
        (0, 1),
        t_eval=t_eval,
        parameters=rv_pressure_parameters,
    )
    activation = activation_model.activation_function(
        (0, 1),
        t_eval=t_eval,
        parameters=activation_parameters,
    )

    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        np.save(
            outdir / "pressure_model.npy",
            {
                "time": time,
                "activation": activation,
                "lv_pressure": lv_pressure,
                "rv_pressure": rv_pressure,
                "lv_pressure_parameters": lv_pressure_parameters,
                "rv_pressure_parameters": rv_pressure_parameters,
                "activation_parameters": activation_parameters,
            },
        )

        postprocess.plot_activation_pressure_function(
            t=time,
            activation=activation,
            lv_pressure=lv_pressure,
            rv_pressure=rv_pressure,
            outdir=outdir,
        )

    plv = dolfin.Constant(0.0)
    prv = dolfin.Constant(0.0)
    problem_parameters["plv"] = plv
    problem_parameters["prv"] = prv

    material = HolzapfelOgden(
        f0=geo.f0,
        n0=geo.n0,
        tau=tau,
        parameters=material_parameters,
    )
    problem = BiVProblem(
        geometry=geo,
        material=material,
        parameters=problem_parameters,
    )

    problem.solve()

    result_filepath = Path(outpath)
    if result_filepath.suffix != ".h5":
        msg = f"Expected output path to be to type HDF5 with suffix .h5, got {result_filepath.suffix}"
        raise OSError(msg)
    result_filepath.parent.mkdir(exist_ok=True)

    collector = postprocess.DataCollector(
        result_filepath,
        problem=problem,
        pressure_parameters={
            "lv": lv_pressure_parameters,
            "rv": rv_pressure_parameters,
        },
        actvation_parameters=activation_parameters,
    )

    solve(
        problem=problem,
        tau=tau,
        activation=activation,
        lv_pressure=lv_pressure,
        rv_pressure=rv_pressure,
        plv=plv,
        prv=prv,
        time=time,
        collector=collector,
        store_freq=1,
    )
