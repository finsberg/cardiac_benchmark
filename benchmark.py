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


def plot_componentwise_displacement(
    loader: DataLoader,
    fname="componentwise_displacement.png",
):
    fname = Path(fname).with_suffix(".png")
    p0 = (0.025, 0.03, 0)
    up0 = loader.deformation_at_point(p0)
    basefname = fname.with_suffix("").as_posix()
    np.save(Path(basefname + "_up0").with_suffix(".npy"), up0)

    p1 = (0, 0.03, 0)
    up1 = loader.deformation_at_point(p1)
    np.save(Path(basefname + "_up1").with_suffix(".npy"), up1)

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(loader.time_stamps, up0[:, 0], label="x")
    ax[0].plot(loader.time_stamps, up0[:, 1], label="y")
    ax[0].plot(loader.time_stamps, up0[:, 2], label="z")
    ax[0].set_ylabel("$u(p_0)$[m]")

    ax[1].plot(loader.time_stamps, up1[:, 0], label="x")
    ax[1].plot(loader.time_stamps, up1[:, 1], label="y")
    ax[1].plot(loader.time_stamps, up1[:, 2], label="z")
    ax[1].set_ylabel("$u(p_1)$[m]")
    ax[1].set_xlabel("Time [s]")

    for axi in ax:
        axi.legend()
        axi.grid()
    fig.savefig(fname)


def plot_volume(loader: DataLoader, fname="volume.png"):
    volumes = loader.cavity_volume()
    fname = Path(fname).with_suffix(".png")
    basefname = fname.with_suffix("").as_posix()
    np.save(Path(basefname + "_volumes").with_suffix(".npy"), volumes)

    fig, ax = plt.subplots()
    ax.plot(loader.time_stamps, volumes)
    ax.set_ylabel("Volume [m^3]")
    ax.set_xlabel("Time [s]")
    ax.grid()
    ax.set_title("Volume throug time")
    fig.savefig(fname)


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
    time = np.arange(0, 1, dt)
    plot_activation_function(t=time)
    _, act = activation_function((0, 1), t_eval=time)

    material = HolzapfelOgden(f0=geo.f0, n0=geo.n0, tau=tau)

    problem = Problem(
        geometry=geo,
        material=material,
        solver_parameters={"verbose": True},
        function_space="P_2",
    )

    problem.parameters["dt"].assign(dt)
    problem.solve()

    result_filepath = Path("results.h5")
    collector = DataCollector(result_filepath, u=problem.u, geometry=geo)

    solve(problem, tau, act, time, collector)


def postprocess():
    geo = get_geometry()

    loader = DataLoader("results.h5", geo)
    loader.to_xdmf("u.xdmf")

    plot_componentwise_displacement(loader, "componentwise_displacement.png")
    plot_volume(loader, "volume.png")


if __name__ == "__main__":
    # main()
    postprocess()
