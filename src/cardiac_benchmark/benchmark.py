import dolfin
import numpy as np
from pathlib import Path
from typing import Dict
from typing import Union

from . import postprocess
from . import pressure_model
from .geometry import EllipsoidGeometry
from .material import HolzapfelOgden
from .problem import Problem

HERE = Path(__file__).absolute().parent

dolfin.parameters["form_compiler"]["quadrature_degree"] = 4
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["representation"] = "uflacs"
dolfin.parameters["form_compiler"]["optimize"] = True


def solve(
    problem,
    tau: dolfin.Constant,
    act: np.ndarray,
    pressure: np.ndarray,
    p: dolfin.Constant,
    time: np.ndarray,
    collector: postprocess.DataCollector,
    store_freq: int = 1,
) -> None:

    for i, (t, a, p_) in enumerate(zip(time, act, pressure)):
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
    alpha_endo: float = -60.0,
    alpha_epi: float = 60.0,
) -> EllipsoidGeometry:
    if not path.is_file():
        geo = EllipsoidGeometry.from_parameters(
            fiber_params={"alpha_endo": alpha_endo, "alpha_epi": alpha_epi},
        )
        geo.save(path)
    return EllipsoidGeometry.from_file(path)


def default_parameters() -> Dict[str, Union[float, str]]:
    return dict(
        rho=1e3,
        kappa=1e6,
        alpha_top=1e5,
        alpha_epi=1e8,
        beta_top=5e3,
        beta_epi=5e3,
        t_sys=0.17,
        t_dias=0.484,
        gamma=0.005,
        a_max=5.0,
        a_min=-30.0,
        sigma_0=1e5,
        a=59.0,
        a_f=18472.0,
        a_fn=216.0,
        a_n=2481.0,
        b=8.023,
        b_f=16.026,
        b_fn=11.436,
        b_n=11.12,
        eta=1e2,
        k=100.0,
        dt=0.001,
        alpha_m=0.2,
        alpha_f=0.4,
        alpha_endo_fiber=-60.0,
        alpha_epi_fiber=60.0,
        outpath="results.h5",
        geometry_path="geometry.h5",
    )


def run(
    rho: float = 1e3,
    kappa: float = 1e6,
    alpha_top: float = 1e5,
    alpha_epi: float = 1e8,
    beta_top: float = 5e3,
    beta_epi: float = 5e3,
    t_sys: float = 0.17,
    t_dias: float = 0.484,
    gamma: float = 0.005,
    a_max: float = 5.0,
    a_min: float = -30.0,
    sigma_0: float = 1e5,
    alpha_pre: float = 20.0,
    alpha_mid: float = 5.0,
    sigma_pre: float = 9332.4,
    sigma_mid: float = 15998.0,
    a: float = 59.0,
    a_f: float = 18472.0,
    a_fn: float = 216.0,
    a_n: float = 2481.0,
    b: float = 8.023,
    b_f: float = 16.026,
    b_fn: float = 11.436,
    b_n: float = 11.12,
    eta: float = 1e2,
    k: float = 100.0,
    dt: float = 0.001,
    alpha_m: float = 0.2,
    alpha_f: float = 0.4,
    alpha_endo_fiber: float = -60.0,
    alpha_epi_fiber: float = 60.0,
    outpath: Union[str, Path] = "results.h5",
    geometry_path: Union[str, Path] = "geometry.h5",
) -> None:
    outdir = Path(outpath).parent
    outdir.mkdir(parents=True, exist_ok=True)

    problem_parameters = Problem.default_parameters()
    problem_parameters["alpha_top"].assign(alpha_top)
    problem_parameters["alpha_epi"].assign(alpha_epi)
    problem_parameters["beta_top"].assign(beta_top)
    problem_parameters["beta_epi"].assign(beta_epi)
    problem_parameters["rho"].assign(rho)
    problem_parameters["dt"].assign(dt)
    problem_parameters["alpha_m"].assign(alpha_m)
    problem_parameters["alpha_f"].assign(alpha_f)

    pressure_parameters = pressure_model.default_parameters()
    pressure_parameters["sigma_0"] = sigma_0
    pressure_parameters["t_sys"] = t_sys
    pressure_parameters["t_dias"] = t_dias
    pressure_parameters["gamma"] = gamma
    pressure_parameters["a_max"] = a_max
    pressure_parameters["a_min"] = a_min
    pressure_parameters["alpha_pre"] = alpha_pre
    pressure_parameters["alpha_mid"] = alpha_mid
    pressure_parameters["sigma_pre"] = sigma_pre
    pressure_parameters["sigma_mid"] = sigma_mid

    material_parameters = HolzapfelOgden.default_parameters()
    material_parameters["eta"].assign(eta)
    material_parameters["a"].assign(a)
    material_parameters["a_f"].assign(a_f)
    material_parameters["a_fn"].assign(a_fn)
    material_parameters["a_n"].assign(a_n)
    material_parameters["b"].assign(b)
    material_parameters["b_f"].assign(b_f)
    material_parameters["b_fn"].assign(b_fn)
    material_parameters["b_n"].assign(b_n)
    material_parameters["k"].assign(k)
    material_parameters["kappa"].assign(kappa)

    tau = dolfin.Constant(0.0)
    time = np.arange(dt, 1, dt)

    t_eval = time - float(problem_parameters["alpha_f"]) * dt
    pm = pressure_model.activation_pressure_function(
        (0, 1),
        t_eval=t_eval,
        parameters=pressure_parameters,
    )
    pm.save(outdir / "pressure_model.npy")
    postprocess.plot_activation_pressure_function(
        t=time,
        act=pm.act,
        pressure=pm.pressure,
        outdir=outdir,
    )

    p = dolfin.Constant(0.0)
    problem_parameters["p"] = p

    geo = get_geometry(
        path=Path(geometry_path),
        alpha_endo=alpha_endo_fiber,
        alpha_epi=alpha_epi_fiber,
    )
    material = HolzapfelOgden(
        f0=geo.f0,
        n0=geo.n0,
        tau=tau,
        parameters=material_parameters,
    )

    problem = Problem(
        geometry=geo,
        material=material,
        function_space="P_1",
        parameters=problem_parameters,
    )
    problem.solve()

    result_filepath = Path(outpath)
    if result_filepath.suffix != ".h5":
        msg = "Expected output path to be to type HDF5 with suffix .h5, got {result_filepath.suffix}"
        raise OSError(msg)
    result_filepath.parent.mkdir(exist_ok=True)
    collector = postprocess.DataCollector(
        result_filepath,
        problem=problem,
        pressure_parameters=pm.parameters,
    )

    solve(
        problem=problem,
        tau=tau,
        act=pm.act,
        pressure=pm.pressure,
        p=p,
        time=time,
        collector=collector,
        store_freq=1,
    )
