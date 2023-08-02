import typing

import dolfinx
import ufl


def heaviside(
    x: ufl.Coefficient,
    k: float,
    use_exp: bool = True,
) -> ufl.Coefficient:
    r"""
    Heaviside function

    .. math::
       \frac{\mathrm{d}}{\mathrm{d}x} \max\{x,0\}

    or

    .. math::
        \frac{1}{1 + e^{-k (x - 1)}}
    """

    if use_exp:
        return 1 / (1 + ufl.exp(-k * (x - 1)))
    else:
        return ufl.conditional(ufl.ge(x, 0.0), 1.0, 0.0)


class HolzapfelOgden:
    r"""
    Viscoelastic version of the Holzapfel and Ogden

    Parameters
    ----------
    f0: dolfinx.fem.Function
        Function representing the direction of the fibers
    s0: dolfinx.fem.Function
        Function representing the direction of the sheets
    tau: dolfinx.fem.Constant
        The active stress
    parameters: Dict[str, float]
        Dictionary with material parameters
    k : float

    Notes
    -----
    Modified version of the original model from Holzapfel and Ogden [2]_.

    The strain energy density function is given by

    .. math::
        \Psi(I_1, I_{4\mathbf{f}_0}, I_{4\mathbf{s}_0}, I_{8\mathbf{f}_0\mathbf{s}_0})
        = \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)
        + \frac{a_f}{2 b_f} \mathcal{H}(I_{4\mathbf{f}_0} - 1)
        \left( e^{ b_f (I_{4\mathbf{f}_0} - 1)_+^2} -1 \right)
        + \frac{a_s}{2 b_s} \mathcal{H}(I_{4\mathbf{s}_0} - 1)
        \left( e^{ b_s (I_{4\mathbf{s}_0} - 1)_+^2} -1 \right)
        + \frac{a_{fs}}{2 b_{fs}} \left( e^{ b_{fs}
        I_{8 \mathbf{f}_0 \mathbf{s}_0}^2} -1 \right)

    where

    .. math::
        (x)_+ = \max\{x,0\}

    and

    .. math::
        \mathcal{H}(x) = \begin{cases}
            1, & \text{if $x > 0$} \\
            0, & \text{if $x \leq 0$}
        \end{cases}

    is the Heaviside function.

        .. [2] Holzapfel, Gerhard A., and Ray W. Ogden.
            "Constitutive modelling of passive myocardium:
            a structurally based framework for material characterization.
            "Philosophical Transactions of the Royal Society of London A:
            Mathematical, Physical and Engineering Sciences 367.1902 (2009):
            3445-3475.
    """

    def __init__(
        self,
        f0: dolfinx.fem.Function,
        s0: dolfinx.fem.Function,
        tau: dolfinx.fem.Constant,
        parameters: typing.Optional[typing.Dict[str, float]] = None,
    ) -> None:
        parameters = parameters or {}
        self.parameters = HolzapfelOgden.default_parameters()
        self.parameters.update(parameters)

        self.f0 = f0
        self.s0 = s0
        self.tau = tau

    @staticmethod
    def default_parameters() -> typing.Dict[str, float]:
        return {
            "a": 59.0,
            "b": 8.023,
            "a_f": 18472.0,
            "b_f": 16.026,
            "a_s": 2481.0,
            "b_s": 11.120,
            "a_fs": 216.0,
            "b_fs": 11.436,
            "kappa": 1e6,
            "eta": 1e2,
            "k": 100.0,
        }

    def W_1(self, I1):
        a = self.parameters["a"]
        b = self.parameters["b"]

        return a / (2.0 * b) * (ufl.exp(b * (I1 - 3)) - 1.0)

    def W_4(self, I4, direction):
        assert direction in ["f", "s"]
        a = self.parameters[f"a_{direction}"]
        b = self.parameters[f"b_{direction}"]

        return (
            a
            / (2.0 * b)
            * heaviside(I4, k=self.parameters["k"])
            * (ufl.exp(b * (I4 - 1) ** 2) - 1.0)
        )

    def W_8(self, I8):
        """
        Cross fiber-sheet contribution.
        """
        a = self.parameters["a_fs"]
        b = self.parameters["b_fs"]

        return a / (2.0 * b) * (ufl.exp(b * I8**2) - 1.0)

    def W_compress(self, J):
        """
        Compressibility contribution
        """
        return 0.25 * self.parameters["kappa"] * (J**2 - 1 - 2 * ufl.ln(J))

    def W_visco(self, E_dot):
        """Viscoelastic contributions"""
        return 0.5 * self.parameters["eta"] * ufl.tr(E_dot * E_dot)

    def Wactive(self, I4f):
        return 0.5 * self.tau * (I4f - 1)

    def strain_energy(self, F):
        """
        Strain-energy density function.
        """

        # Invariants
        C = F.T * F
        J = ufl.det(F)

        I1 = pow(J, -2 / 3) * ufl.tr(C)
        breakpoint()
        I4f = ufl.inner(self.f0, C * self.f0)
        I4s = ufl.inner(self.s0, C * self.s0)
        I8fs = ufl.inner(self.f0, C * self.s0)

        # Compressibility
        Wcompress = self.W_compress(J)

        # Active stress
        Wactive = self.Wactive(I4f)

        W1 = self.W_1(I1)
        W4f = self.W_4(I4f, "f")
        W4s = self.W_4(I4s, "s")
        W8fs = self.W_8(I8fs)

        W = W1 + W4f + W4s + W8fs + Wactive + Wcompress
        return W
