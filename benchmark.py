import math
from pathlib import Path

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

from geometry import EllipsoidGeometry
from material import HolzapfelOgden
from postprocess import DataCollector
from postprocess import DataLoader
from problem import Problem


dolfin.parameters["form_compiler"]["quadrature_degree"] = 4
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


def plot_componentwise_displacement(loader: DataLoader):

    p0 = (0.025, 0.03, 0)
    up0 = loader.deformation_at_point(p0)

    p1 = (0, 0.03, 0)
    up1 = loader.deformation_at_point(p1)

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(loader.time_stamps, up0[:, 0], label="x")
    ax[0].plot(loader.time_stamps, up0[:, 1], label="y")
    ax[0].plot(loader.time_stamps, up0[:, 2], label="z")
    ax[0].legend()
    ax[0].set_ylabel("$u(p_0)$[m]")

    ax[1].plot(loader.time_stamps, up1[:, 0], label="x")
    ax[1].plot(loader.time_stamps, up1[:, 1], label="y")
    ax[1].plot(loader.time_stamps, up1[:, 2], label="z")
    ax[1].legend()
    ax[1].set_ylabel("$u(p_1)$[m]")
    ax[1].set_xlabel("Time [s]")
    fig.savefig("componentwise_displacement.png")


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


def main():
    path = Path("geometry.h5")
    if not path.is_file():
        geo = EllipsoidGeometry.from_parameters()
        geo.save(path)
    geo = EllipsoidGeometry.from_file(path)

    tau = dolfin.Constant(0.0)
    dt = 0.001
    time = np.arange(0, 1, dt)
    plot_activation_function(t=time)
    _, act = activation_function((0, 1), t_eval=time)

    material = HolzapfelOgden(f0=geo.f0, s0=geo.s0, tau=tau)

    problem = Problem(geometry=geo, material=material)
    problem.parameters["dt"].assign(dt)
    problem.solve()

    result_filepath = Path("results.h5")
    collector = DataCollector(result_filepath, u=problem.u)

    solve(problem, tau, act, time, collector)


def postprocess():

    loader = DataLoader("results.h5")
    loader.to_xdmf("u.xdmf")

    plot_componentwise_displacement(loader)


if __name__ == "__main__":
    main()
    postprocess()
