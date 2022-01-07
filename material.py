import typing

import dolfin


def heaviside(x, k=100):
    r"""
    Heaviside function

    .. math::
       \frac{\mathrm{d}}{\mathrm{d}x} \max\{x,0\}

    or

    .. math::
        \frac{1}{1 + e^{-k (x - 1)}}
    """

    # return dolfin.conditional(dolfin.ge(x, 0.0), 1.0, 0.0)
    return 1 / (1 + dolfin.exp(-k * (x - 1)))


def I1(F):
    J = dolfin.det(F)
    C = F.T * F
    return pow(J, -2 / 3) * dolfin.tr(C)


def I4(F, a0):
    return dolfin.inner(F * a0, F * a0)


def I8(F, a0, b0):
    return dolfin.inner(F * a0, F * b0)


class HolzapfelOgden:
    def __init__(
        self,
        f0: dolfin.Function,
        n0: dolfin.Function,
        tau: dolfin.Constant = dolfin.Constant(0.0),
        parameters: typing.Optional[typing.Dict[str, dolfin.Constant]] = None,
    ) -> None:

        parameters = parameters or {}
        self.parameters = HolzapfelOgden.default_parameters()
        self.parameters.update(parameters)

        self.f0 = f0
        self.n0 = n0
        self.tau = tau

    @staticmethod
    def default_parameters() -> typing.Dict[str, dolfin.Constant]:
        return {
            "a": dolfin.Constant(59.0),
            "b": dolfin.Constant(8.023),
            "a_f": dolfin.Constant(18472.0),
            "b_f": dolfin.Constant(16.026),
            "a_n": dolfin.Constant(2481.0),
            "b_n": dolfin.Constant(11.120),
            "a_fn": dolfin.Constant(216.0),
            "b_fn": dolfin.Constant(11.436),
            "kappa": dolfin.Constant(1e6),
            "eta": dolfin.Constant(1e2),
        }

    def W_1(self, I1):
        a = self.parameters["a"]
        b = self.parameters["b"]

        return a / (2.0 * b) * (dolfin.exp(b * (I1 - 3)) - 1.0)

    def W_4(self, I4, direction):
        assert direction in ["f", "n"]
        a = self.parameters[f"a_{direction}"]
        b = self.parameters[f"b_{direction}"]

        return a / (2.0 * b) * heaviside(I4) * (dolfin.exp(b * (I4 - 1) ** 2) - 1.0)

    def W_8(self, I8):
        """
        Cross fiber-sheet contribution.
        """
        a = self.parameters["a_fn"]
        b = self.parameters["b_fn"]

        return a / (2.0 * b) * (dolfin.exp(b * I8 ** 2) - 1.0)

    def W_compress(self, J):
        """
        Compressibility contribution
        """
        return 0.25 * self.parameters["kappa"] * (J ** 2 - 1 - 2 * dolfin.ln(J))

    def W_visco(self, E_dot):
        """Viscoelastic contributions"""
        return 0.5 * self.parameters["eta"] * dolfin.tr(E_dot * E_dot)

    def Wactive(self, I4f):
        return dolfin.Constant(0.5) * self.tau * (I4f - 1)

    def strain_energy(self, F):
        """
        Strain-energy density function.
        """

        # Invariants
        J = dolfin.det(F)
        C = F.T * F
        I1 = pow(J, -2 / 3) * dolfin.tr(C)
        I4f = dolfin.inner(F * self.f0, F * self.f0)
        I4s = dolfin.inner(F * self.n0, F * self.n0)
        I8fs = dolfin.inner(F * self.f0, F * self.n0)

        # Compressibility
        Wcompress = self.W_compress(J)

        # Active stress
        Wactive = self.Wactive(I4f)

        W1 = self.W_1(I1)
        W4f = self.W_4(I4f, "f")
        W4s = self.W_4(I4s, "n")
        W8fs = self.W_8(I8fs)

        W = W1 + W4f + W4s + W8fs + Wcompress + Wactive
        return W
