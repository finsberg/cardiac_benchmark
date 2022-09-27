import math
import typing

import numpy as np
import scipy.integrate


def default_parameters():
    return dict(
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
    )


def activation_pressure_function(
    t_span: typing.Tuple[float, float],
    t_eval: typing.Optional[np.ndarray] = None,
    parameters: typing.Optional[typing.Dict[str, float]] = None,
):

    params = default_parameters()
    if parameters is not None:
        params.update(parameters)

    f = (
        lambda t: 0.25
        * (1 + math.tanh((t - params["t_sys"]) / params["gamma"]))
        * (1 - math.tanh((t - params["t_dias"]) / params["gamma"]))
    )
    a = lambda t: params["a_max"] * f(t) + params["a_min"] * (1 - f(t))

    f_pre = lambda t: 0.5 * (1 - math.tanh((t - params["t_dias"]) / params["gamma"]))
    b = lambda t: a(t) + params["alpha_pre"] * f_pre(t) + params["alpha_mid"]

    def rhs(t, state):
        tau, p = state
        return [
            -abs(a(t)) * tau + params["sigma_0"] * max(a(t), 0),
            -abs(b(t)) * p
            + params["sigma_mid"] * max(b(t), 0)
            + params["sigma_pre"] * max(f_pre(t), 0),
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
