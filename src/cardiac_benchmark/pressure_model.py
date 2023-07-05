import math
import pprint
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import scipy.integrate


def default_parameters_benchmark1():
    return dict(
        t_sys_pre=0.17,
        t_dias_pre=0.484,
        gamma=0.005,
        a_max=5.0,
        a_min=-30.0,
        alpha_pre=5.0,
        alpha_mid=1.0,
        sigma_pre=7000.0,
        sigma_mid=16000.0,
    )


def default_lv_parameters_benchmark2():
    return dict(
        t_sys_pre=0.17,
        t_dias_pre=0.484,
        gamma=0.005,
        a_max=5.0,
        a_min=-30.0,
        alpha_pre=5.0,
        alpha_mid=15.0,
        sigma_pre=12000.0,
        sigma_mid=16000.0,
    )


def default_rv_parameters_benchmark2():
    return dict(
        t_sys_pre=0.17,
        t_dias_pre=0.484,
        gamma=0.005,
        a_max=5.0,
        a_min=-30.0,
        alpha_pre=1.0,
        alpha_mid=10.0,
        sigma_pre=3000.0,
        sigma_mid=4000.0,
    )


def pressure_function(
    t_span: Tuple[float, float],
    parameters: Dict[str, float],
    t_eval: Optional[np.ndarray] = None,
) -> np.ndarray:
    print(f"Solving pressure model with parameters: {pprint.pformat(parameters)}")

    f = (
        lambda t: 0.25
        * (1 + math.tanh((t - parameters["t_sys_pre"]) / parameters["gamma"]))
        * (1 - math.tanh((t - parameters["t_dias_pre"]) / parameters["gamma"]))
    )
    a = lambda t: parameters["a_max"] * f(t) + parameters["a_min"] * (1 - f(t))

    f_pre = lambda t: 0.5 * (
        1 - math.tanh((t - parameters["t_dias_pre"]) / parameters["gamma"])
    )
    b = lambda t: a(t) + parameters["alpha_pre"] * f_pre(t) + parameters["alpha_mid"]

    def rhs(t, p):
        return (
            -abs(b(t)) * p
            + parameters["sigma_mid"] * max(b(t), 0)
            + parameters["sigma_pre"] * max(f_pre(t), 0)
        )

    res = scipy.integrate.solve_ivp(
        rhs,
        t_span,
        [0.0],
        t_eval=t_eval,
        method="Radau",
    )
    return res.y.squeeze()
