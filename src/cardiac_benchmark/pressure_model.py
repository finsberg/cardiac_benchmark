import math
import pprint
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import scipy.integrate


def default_parameters():
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


def pressure_function(
    t_span: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    parameters: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    params = default_parameters()
    if parameters is not None:
        params.update(parameters)

    print(f"Solving pressure model with parameters: {pprint.pformat(params)}")

    f = (
        lambda t: 0.25
        * (1 + math.tanh((t - params["t_sys_pre"]) / params["gamma"]))
        * (1 - math.tanh((t - params["t_dias_pre"]) / params["gamma"]))
    )
    a = lambda t: params["a_max"] * f(t) + params["a_min"] * (1 - f(t))

    f_pre = lambda t: 0.5 * (
        1 - math.tanh((t - params["t_dias_pre"]) / params["gamma"])
    )
    b = lambda t: a(t) + params["alpha_pre"] * f_pre(t) + params["alpha_mid"]

    def rhs(t, p):
        return (
            -abs(b(t)) * p
            + params["sigma_mid"] * max(b(t), 0)
            + params["sigma_pre"] * max(f_pre(t), 0)
        )

    res = scipy.integrate.solve_ivp(
        rhs,
        t_span,
        [0.0],
        t_eval=t_eval,
        method="Radau",
    )
    return res.y.squeeze()
