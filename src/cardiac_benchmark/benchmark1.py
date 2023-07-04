import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import dolfin
import numpy as np

from . import activation_model
from . import postprocess
from . import pressure_model
from .geometry import LVGeometry
from .material import HolzapfelOgden
from .problem import LVProblem

HERE = Path(__file__).absolute().parent
logger = logging.getLogger(__name__)

dolfin.parameters["form_compiler"]["quadrature_degree"] = 4
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["representation"] = "uflacs"
dolfin.parameters["form_compiler"]["optimize"] = True


def solve(
    problem,
    tau: dolfin.Constant,
    activation: np.ndarray,
    pressure: np.ndarray,
    p: dolfin.Constant,
    time: np.ndarray,
    collector: postprocess.DataCollector,
    store_freq: int = 1,
) -> None:
    for i, (t, a, p_) in enumerate(zip(time, activation, pressure)):
        dolfin.info(f"{i}: Solving for time {t:.3f} with tau = {a} and pressure = {p_}")

        tau.assign(a)
        p.assign(p_)
        converged = problem.solve()
        if not converged:
            raise RuntimeError

        if i % store_freq == 0:
            dolfin.info("Store solution")
            collector.store(t)


def get_geometry(
    path: Path,
    mesh_parameters: Optional[Dict[str, float]] = None,
    fiber_parameters: Optional[Dict[str, Union[float, str]]] = None,
) -> LVGeometry:
    if not path.is_file():
        mesh_parameters = _update_parameters(
            LVGeometry.default_mesh_parameters(),
            mesh_parameters,
        )
        fiber_parameters = _update_parameters(
            LVGeometry.default_fiber_parameters(),
            fiber_parameters,
        )
        geo = LVGeometry.from_parameters(
            fiber_params=fiber_parameters,
            mesh_params=mesh_parameters,
        )
        geo.save(path)
    return LVGeometry.from_file(path)


def default_parameters():
    return dict(
        problem_parameters=LVProblem.default_parameters(),
        activation_parameters=activation_model.default_parameters(),
        pressure_parameters=pressure_model.default_parameters(),
        material_parameters=HolzapfelOgden.default_parameters(),
        mesh_parameters=LVGeometry.default_mesh_parameters(),
        fiber_parameters=LVGeometry.default_fiber_parameters(),
        outpath="results.h5",
        geometry_path="geometry.h5",
    )


def _update_parameters(
    _par: Dict[str, Any],
    par: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if par is None:
        par = {}
    for key, value in par.items():
        if key not in _par:
            logger.warning(f"Invalid key {key}")
            continue

        if isinstance(_par[key], dolfin.Constant):
            _par[key].assign(value)
        else:
            _par[key] = value
    return _par


def run(
    problem_parameters: Optional[Dict[str, Union[float, dolfin.Constant]]] = None,
    activation_parameters: Optional[Dict[str, float]] = None,
    pressure_parameters: Optional[Dict[str, float]] = None,
    material_parameters: Optional[Dict[str, Union[float, dolfin.Constant]]] = None,
    mesh_parameters: Optional[Dict[str, float]] = None,
    fiber_parameters: Optional[Dict[str, Union[float, str]]] = None,
    zero_pressure: bool = False,
    zero_activation: bool = False,
    outpath: Union[str, Path] = "results.h5",
    geometry_path: Union[str, Path] = "geometry.h5",
) -> None:
    outdir = Path(outpath).parent
    outdir.mkdir(parents=True, exist_ok=True)

    problem_parameters = _update_parameters(
        LVProblem.default_parameters(),
        problem_parameters,
    )
    pressure_parameters = _update_parameters(
        pressure_model.default_parameters(),
        pressure_parameters,
    )
    activation_parameters = _update_parameters(
        activation_model.default_parameters(),
        activation_parameters,
    )
    material_parameters = _update_parameters(
        HolzapfelOgden.default_parameters(),
        material_parameters,
    )
    mesh_parameters = _update_parameters(
        LVGeometry.default_mesh_parameters(),
        mesh_parameters,
    )
    fiber_parameters = _update_parameters(
        LVGeometry.default_fiber_parameters(),
        fiber_parameters,
    )

    dt = float(problem_parameters["dt"])
    tau = dolfin.Constant(0.0)
    time = np.arange(dt, 1, dt)

    t_eval = time - float(problem_parameters["alpha_f"]) * dt
    pressure = pressure_model.pressure_function(
        (0, 1),
        t_eval=t_eval,
        parameters=pressure_parameters,
    )
    if zero_pressure:
        # We set the pressure to zero
        pressure[:] = 0.0

    activation = activation_model.activation_function(
        (0, 1),
        t_eval=t_eval,
        parameters=activation_parameters,
    )
    if zero_activation:
        activation[:] = 0.0

    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        np.save(
            outdir / "pressure_model.npy",
            {
                "time": time,
                "activation": activation,
                "pressure": pressure,
                "pressure_parameters": pressure_parameters,
                "activation_parameters": activation_parameters,
            },
        )

        postprocess.plot_activation_pressure_function(
            t=time,
            activation=activation,
            pressure=pressure,
            outdir=outdir,
        )

    p = dolfin.Constant(0.0)
    problem_parameters["p"] = p

    geo = get_geometry(
        path=Path(geometry_path),
        fiber_parameters=fiber_parameters,
        mesh_parameters=mesh_parameters,
    )
    material = HolzapfelOgden(
        f0=geo.f0,
        n0=geo.n0,
        tau=tau,
        parameters=material_parameters,
    )
    problem = LVProblem(
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
        pressure_parameters=pressure_parameters,
        actvation_parameters=activation_parameters,
    )

    solve(
        problem=problem,
        tau=tau,
        activation=activation,
        pressure=pressure,
        p=p,
        time=time,
        collector=collector,
        store_freq=1,
    )
