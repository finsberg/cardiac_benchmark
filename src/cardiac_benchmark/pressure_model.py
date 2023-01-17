import math
import pprint
from pathlib import Path
from typing import Dict
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import scipy.integrate


class PressureActivationSolution(NamedTuple):
    time: np.ndarray
    state: np.ndarray
    parameters: Dict[str, float]

    @property
    def act(self) -> np.ndarray:
        return self.state[0, :]

    @property
    def pressure(self) -> np.ndarray:
        return self.state[1, :]

    def save(self, fname: Union[Path, str]) -> None:
        np.save(
            fname,
            {
                "time": self.time,
                "state": self.state,
                "parameters": self.parameters,
            },
        )


def default_parameters():
    return dict(
        t_sys=0.17,
        t_dias=0.484,
        gamma=0.005,
        a_max=5.0,
        a_min=-30.0,
        sigma_0=1e5,
        alpha_pre=20.0,
        alpha_mid=5.0,
        sigma_pre=9332.4,
        sigma_mid=15998.0,
    )


def activation_pressure_function(
    t_span: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    parameters: Optional[Dict[str, float]] = None,
) -> PressureActivationSolution:

    params = default_parameters()
    if parameters is not None:
        params.update(parameters)

    print(f"Solving pressure model with parameters: {pprint.pformat(params)}")

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
    return PressureActivationSolution(
        time=res.t,
        state=res.y.squeeze(),
        parameters=params,
    )
