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


def activation_function(
    t_span,
    t_eval=None,
    t_sys=0.17,
    t_dias=0.484,
    gamma=0.005,
    a_max=5.0,
    a_min=-30.0,
    sigma_0=1e5,
):

    f = (
        lambda t: 0.25
        * (1 + math.tanh((t - t_sys) / gamma))
        * (1 - math.tanh((t - t_dias) / gamma))
    )
    a = lambda t: a_max * f(t) + a_min * (1 - f(t))

    def rhs(t, tau):
        return -abs(a(t)) * tau + sigma_0 * max(a(t), 0)

    res = scipy.integrate.solve_ivp(rhs, t_span, [0.0], t_eval=t_eval)

    return (res.t, res.y.squeeze())


def plot_activation_function(t):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(*activation_function(t_span=(0, 1), t_eval=t))
    ax.set_title("Activation fuction \u03C4(t)")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_xlabel("Time [s]")
    fig.savefig("activation_function.png")


def solve(problem, tau, act, time, collector):

    dt = float(problem.parameters["dt"])

    for t, a in zip(time, act):
        dolfin.info(f"Solving for time {t:.3f} with tau = {a}")

        converged = False
        target_tau = a
        prev_tau = float(tau)
        num_crash = 0

        if not math.isclose(float(tau), target_tau):
            problem.parameters["dt"].assign(dt)
            while not converged and not math.isclose(float(tau), target_tau):
                print(f"Try a = {a}")
                tau.assign(a)

                converged = problem.solve()

                if converged:
                    num_crash = 0
                    prev_tau = a
                    a = target_tau

                else:
                    a = prev_tau + (a - prev_tau) / 2
                    tau.assign(a)
                    problem.parameters["dt"].assign(problem.parameters["dt"] * 0.5)
                    num_crash += 1

                if num_crash > 10:
                    raise RuntimeError

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

    # Newmark beta method
    # parameters["alpha_m"] = dolfin.Constant(0.0)
    # parameters["alpha_f"] = dolfin.Constant(0.0)

    time = np.arange(dt, 1, dt)
    plot_activation_function(t=time)
    _, act = activation_function(
        (0, 1),
        t_eval=time - float(parameters["alpha_f"]) * dt,
    )

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
    collector = DataCollector(result_filepath, u=problem.u, geometry=geo)

    solve(problem, tau, act, time, collector)


def postprocess():
    loader = DataLoader("results.h5")
    loader.postprocess_all()
    loader.compare_results()


if __name__ == "__main__":
    main()
    postprocess()
