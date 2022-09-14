import math
from pathlib import Path

import dolfin
import numpy as np
import scipy.integrate

from geometry import EllipsoidGeometry
from material import HolzapfelOgden
from postprocess import DataCollector
from postprocess import DataLoader
from problem import Problem


dolfin.parameters["form_compiler"]["quadrature_degree"] = 4
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["representation"] = "uflacs"
flags = ["-O3", "-march=native"]
dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
# TODO: Should we add more compiler flags?


def activation_pressure_function(
    t_span,
    t_eval=None,
    t_sys=0.17,
    t_dias=0.484,
    gamma=0.005,
    a_max=5.0,
    a_min=-30.0,
    sigma_0=1e5,
    alpha_pre=5.0,
    alpha_mid=20.0,
    sigma_pre=9332.4,
    sigma_mid=15998.0,
):

    f = (
        lambda t: 0.25
        * (1 + math.tanh((t - t_sys) / gamma))
        * (1 - math.tanh((t - t_dias) / gamma))
    )
    a = lambda t: a_max * f(t) + a_min * (1 - f(t))

    f_pre = lambda t: 0.5 * (1 - math.tanh((t - t_dias) / gamma))
    b = lambda t: a(t) + alpha_pre * f_pre(t) + alpha_mid

    def rhs(t, state):
        tau, p = state
        return [
            -abs(a(t)) * tau + sigma_0 * max(a(t), 0),
            -abs(b(t)) * p + sigma_mid * max(b(t), 0) + sigma_pre * max(f_pre(t), 0),
        ]

    res = scipy.integrate.solve_ivp(
        rhs,
        t_span,
        [0.0, 0.0],
        t_eval=t_eval,
        method="Radau",
    )

    return (res.t, res.y.squeeze())


def plot_activation_pressure_function(t):
    import matplotlib.pyplot as plt

    t, state = activation_pressure_function(t_span=(0, 1), t_eval=t)

    act = state[0, :]
    pressure = state[1, :]

    fig, ax = plt.subplots()
    ax.plot(t, act)
    ax.set_title("Activation fuction \u03C4(t)")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_xlabel("Time [s]")
    fig.savefig("activation_function.png")

    fig, ax = plt.subplots()
    ax.plot(t, pressure)
    ax.set_title("Pressure fuction p(t)")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_xlabel("Time [s]")
    fig.savefig("pressure_function.png")


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


def main():
    geo = get_geometry()

    tau = dolfin.Constant(0.0)

    dt = 0.001
    parameters = Problem.default_parameters()

    time = np.arange(dt, 1, dt)
    plot_activation_pressure_function(t=time)

    _, state = activation_pressure_function(
        (0, 1),
        t_eval=time - float(parameters["alpha_f"]) * dt,
    )
    act = state[0, :]
    pressure = state[1, :]
    p = dolfin.Constant(0.0)
    parameters["p"] = p

    material = HolzapfelOgden(f0=geo.f0, n0=geo.n0, tau=tau)

    problem = Problem(
        geometry=geo,
        material=material,
        function_space="P_1",
        parameters=parameters,
    )

    problem.parameters["dt"].assign(dt)
    problem.solve()

    result_filepath = Path("results.h5")
    collector = DataCollector(result_filepath, problem=problem)

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


def postprocess():
    loader = DataLoader("results.h5")
    loader.postprocess_all()
    # loader.compare_results()


if __name__ == "__main__":
    # main()
    postprocess()
