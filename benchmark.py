import math
from pathlib import Path

import dolfin
import numpy as np
import scipy.integrate

from geometry import EllipsoidGeometry
from material import HolzapfelOgden
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


def plot_activation_function():
    import matplotlib.pyplot as plt

    t = np.linspace(0, 1, 200)
    fig, ax = plt.subplots()
    ax.plot(*activation_function(t_span=(0, 1), t_eval=t))
    ax.set_title("Activation fuction \u03C4(t)")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_xlabel("Time [s]")
    plt.show()


def main():
    path = Path("geometry.h5")
    if not path.is_file():
        geo = EllipsoidGeometry.from_parameters()
        geo.save(path)
    geo = EllipsoidGeometry.from_file(path)

    tau = dolfin.Constant(0.0)
    dt = 0.01
    time = np.arange(0, 1, dt)
    _, act = activation_function((0, 1), t_eval=time)

    material = HolzapfelOgden(f0=geo.f0, s0=geo.s0, tau=tau)

    problem = Problem(geometry=geo, material=material)
    problem.parameters["dt"].assign(dt)
    problem.solve()

    u_file = dolfin.XDMFFile(geo.mesh.mpi_comm(), "u.xdmf")

    for t, a in zip(time, act):
        dolfin.info(f"Solveing for time {t:.2f}")
        tau.assign(a)
        problem.solve()
        u_file.write(problem.u, t)


if __name__ == "__main__":
    main()
