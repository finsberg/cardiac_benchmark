import math
import pprint
from enum import Enum
from pathlib import Path
from typing import Dict
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import scipy.integrate


class Pressure(str, Enum):
    bestel = "bestel"
    zero_pressure = "zero_pressure"
    zero_active = "zero_active"


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
        t_sys=0.16,
        t_dias=0.484,
        gamma=0.005,
        a_max=5.0,
        a_min=-30.0,
        sigma_0=150e3,
    )


def activation_function(
    t_span: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    parameters: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    params = default_parameters()
    if parameters is not None:
        params.update(parameters)

    print(f"Solving active stress model with parameters: {pprint.pformat(params)}")

    f = (
        lambda t: 0.25
        * (1 + math.tanh((t - params["t_sys"]) / params["gamma"]))
        * (1 - math.tanh((t - params["t_dias"]) / params["gamma"]))
    )
    a = lambda t: params["a_max"] * f(t) + params["a_min"] * (1 - f(t))

    def rhs(t, tau):
        return -abs(a(t)) * tau + params["sigma_0"] * max(a(t), 0)

    res = scipy.integrate.solve_ivp(
        rhs,
        t_span,
        [0.0],
        t_eval=t_eval,
        method="Radau",
    )
    return res.y.squeeze()
