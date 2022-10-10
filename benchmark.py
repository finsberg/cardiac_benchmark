from pathlib import Path

import dolfin
import numpy as np

import pressure_model
from geometry import EllipsoidGeometry
from material import HolzapfelOgden
from postprocess import DataCollector
from postprocess import DataLoader
from problem import Problem


dolfin.parameters["form_compiler"]["quadrature_degree"] = 4
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["representation"] = "uflacs"
# flags = ["-O3", "-march=native"]
# dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
# TODO: Should we add more compiler flags?


def solve(problem, tau, act, pressure, p, time, collector, store_freq: int = 1):

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


def get_geometry():
    path = Path("geometry.h5")
    if not path.is_file():
        geo = EllipsoidGeometry.from_parameters()
        geo.save(path)
    return EllipsoidGeometry.from_file(path)


def run_benchmark(
    alpha_epi: float = 1e8,
    eta: float = 1e2,
    a_f: float = 18472.0,
    sigma_0: float = 1e5,
    outpath: str = "results.h5",
):

    problem_parameters = Problem.default_parameters()
    pressure_parameters = pressure_model.default_parameters()
    material_parameters = HolzapfelOgden.default_parameters()

    problem_parameters["alpha_epi"].assign(alpha_epi)
    material_parameters["eta"].assign(eta)
    material_parameters["a_f"].assign(a_f)
    pressure_parameters["sigma_0"] = sigma_0

    tau = dolfin.Constant(0.0)
    dt = 0.001
    time = np.arange(dt, 1, dt)

    pressure_model.plot_activation_pressure_function(t=time)

    _, state = pressure_model.activation_pressure_function(
        (0, 1),
        t_eval=time - float(problem_parameters["alpha_f"]) * dt,
        parameters=pressure_parameters,
    )
    act = state[0, :]
    pressure = state[1, :]
    p = dolfin.Constant(0.0)
    problem_parameters["p"] = p

    geo = get_geometry()
    material = HolzapfelOgden(f0=geo.f0, n0=geo.n0, tau=tau)

    problem = Problem(
        geometry=geo,
        material=material,
        function_space="P_1",
        parameters=problem_parameters,
    )

    problem.parameters["dt"].assign(dt)
    problem.solve()

    result_filepath = Path(outpath)
    if result_filepath.suffix != ".h5":
        msg = "Expected output path to be to type HDF5 with suffix .h5, got {result_filepath.suffix}"
        raise OSError(msg)
    result_filepath.parent.mkdir(exist_ok=True)
    collector = DataCollector(
        result_filepath,
        problem=problem,
        pressure_parameters=pressure_parameters,
    )

    solve(
        problem=problem,
        tau=tau,
        act=act,
        pressure=pressure,
        p=p,
        time=time,
        collector=collector,
        store_freq=10,
    )


def main():
    outpath = "results.h5"
    run_benchmark(outpath=outpath)
    loader = DataLoader(outpath)
    loader.postprocess_all()


if __name__ == "__main__":
    main()
