import typing

import dolfin


def subplus(x):
    r"""
    Ramp function
    .. math::
       \max\{x,0\}
    """

    return dolfin.conditional(dolfin.ge(x, 0.0), x, 0.0)


def heaviside(x):
    r"""
    Heaviside function
    .. math::
       \frac{\mathrm{d}}{\mathrm{d}x} \max\{x,0\}
    """

    return dolfin.conditional(dolfin.ge(x, 0.0), 1.0, 0.0)


def I1(F):
    J = dolfin.det(F)
    C = F.T * F
    return pow(J, -2 / 3) * dolfin.tr(C)


def I4(F, a0):
    return dolfin.inner(F * a0, F * a0)


def I8(F, a0, b0):
    return dolfin.inner(F * a0, F * b0)


class HolzapfelOgden:
    r"""
    Orthotropic model by Holzapfel and Ogden
    .. math::
       \mathcal{W}(I_1, I_{4f_0})
       = \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)
       + \frac{a_f}{2 b_f} \left( e^{ b_f (I_{4f_0} - 1)_+^2} -1 \right)
       + \frac{a_s}{2 b_s} \left( e^{ b_s (I_{4s_0} - 1)_+^2} -1 \right)
       + \frac{a_fs}{2 b_fs} \left( e^{ b_fs I_{8fs}^2} -1 \right)
    where
    .. math::
       (\cdot)_+ = \max\{x,0\}
    .. rubric:: Reference
    [1] Holzapfel, Gerhard A., and Ray W. Ogden.
    "Constitutive modelling of passive myocardium:
    a structurally based framework for material characterization.
    "Philosophical Transactions of the Royal Society of London A:
    Mathematical, Physical and Engineering Sciences 367.1902 (2009): 3445-3475.
    """

    def __init__(
        self,
        f0: dolfin.Function,
        s0: dolfin.Function,
        tau: dolfin.Constant = dolfin.Constant(0.0),
        parameters: typing.Optional[typing.Dict[str, dolfin.Constant]] = None,
    ) -> None:

        parameters = parameters or {}
        self.parameters = HolzapfelOgden.default_parameters()
        self.parameters.update(parameters)

        self.f0 = f0
        self.s0 = s0
        self.tau = tau

    @staticmethod
    def default_parameters() -> typing.Dict[str, dolfin.Constant]:
        """
        Default matereial parameter for the Holzapfel Ogden model
        Taken from Table 1 row 3 of [1]
        """

        return {
            "a": dolfin.Constant(59.0),
            "b": dolfin.Constant(8.023),
            "a_f": dolfin.Constant(18472.0),
            "b_f": dolfin.Constant(16.026),
            "a_s": dolfin.Constant(2481.0),
            "b_s": dolfin.Constant(11.120),
            "a_fs": dolfin.Constant(216.0),
            "b_fs": dolfin.Constant(11.436),
            "kappa": dolfin.Constant(1e6),
            "eta": dolfin.Constant(1e2),
        }

    def W_1(self, I1, diff=0, *args, **kwargs):
        r"""
        Isotropic contribution.
        If `diff = 0`, return
        .. math::
           \frac{a}{2 b} \left( e^{ b (I_1 - 3)}  -1 \right)
        If `diff = 1`, return
        .. math::
           \frac{a}{b} e^{ b (I_1 - 3)}
        If `diff = 2`, return
        .. math::
           \frac{a b}{2}  e^{ b (I_1 - 3)}
        """

        a = self.parameters["a"]
        b = self.parameters["b"]

        if diff == 0:
            if float(a) > dolfin.DOLFIN_EPS:
                if float(b) > dolfin.DOLFIN_EPS:
                    return a / (2.0 * b) * (dolfin.exp(b * (I1 - 3)) - 1.0)
                else:
                    return a / 2.0 * (I1 - 3)
            else:
                return 0.0
        elif diff == 1:
            return a / 2.0 * dolfin.exp(b * (I1 - 3))
        elif diff == 2:
            return a * b / 2.0 * dolfin.exp(b * (I1 - 3))

    def W_4(self, I4, direction, diff=0):
        r"""
        Anisotropic contribution.
        If `diff = 0`, return
        .. math::
           \frac{a_f}{2 b_f} \left( e^{ b_f (I_{4f_0} - 1)_+^2} -1 \right)
        If `diff = 1`, return
        .. math::
           a_f (I_{4f_0} - 1)_+ e^{ b_f (I_{4f_0} - 1)^2}
        If `diff = 2`, return
        .. math::
           a_f h(I_{4f_0} - 1) (1 + 2b(I_{4f_0} - 1))
           e^{ b_f (I_{4f_0} - 1)_+^2}
        where
        .. math::
           h(x) = \frac{\mathrm{d}}{\mathrm{d}x} \max\{x,0\}
        is the Heaviside function.
        """
        assert direction in ["f", "s", "n"]
        a = self.parameters[f"a_{direction}"]
        b = self.parameters[f"b_{direction}"]

        if I4 == 0:
            return 0

        if diff == 0:
            if float(a) > dolfin.DOLFIN_EPS:
                if float(b) > dolfin.DOLFIN_EPS:
                    return a / (2.0 * b) * (dolfin.exp(b * subplus(I4 - 1) ** 2) - 1.0)
                else:
                    return a / 2.0 * subplus(I4 - 1) ** 2
            else:
                return 0.0

        elif diff == 1:
            return a * subplus(I4 - 1) * dolfin.exp(b * pow(I4 - 1, 2))
        elif diff == 2:
            return (
                a
                * heaviside(I4 - 1)
                * (1 + 2.0 * b * pow(I4 - 1, 2))
                * dolfin.exp(b * pow(I4 - 1, 2))
            )

    def W_8(self, I8, *args, **kwargs):
        """
        Cross fiber-sheet contribution.
        """
        a = self.parameters["a_fs"]
        b = self.parameters["b_fs"]

        if float(a) > dolfin.DOLFIN_EPS:
            if float(b) > dolfin.DOLFIN_EPS:
                return a / (2.0 * b) * (dolfin.exp(b * I8 ** 2) - 1.0)
            else:
                return a / 2.0 * I8 ** 2
        else:
            return 0.0

    def W_compress(self, J):
        """
        Compressibility contribution
        """
        return 0.25 * self.parameters["kappa"] * (J ** 2 - 1 - 2 * dolfin.ln(J))

    def W_visco(self, F, F_dot):
        """Viscoelastic contributions

        Parameters
        ----------
        F : [type]
            [description]
        """
        E_dot = 0.5 * (F.T * F_dot + F_dot.T * F)
        E_dot2 = E_dot.T * E_dot  # FIXME: Is this correct?
        return 0.5 * self.parameters["eta"] * dolfin.tr(E_dot2)

    def Wactive(self, I4f, diff=0):
        if diff == 1:
            return self.tau
        return dolfin.Constant(0.5) * self.tau * (I4f - 1)

    def strain_energy(self, F, F_dot):
        """
        Strain-energy density function.
        """

        # Invariants
        J = dolfin.det(F)
        C = F.T * F
        I1 = pow(J, -2 / 3) * dolfin.tr(C)
        I4f = dolfin.inner(F * self.f0, F * self.f0)
        I4s = dolfin.inner(F * self.s0, F * self.s0)
        I8fs = dolfin.inner(F * self.f0, F * self.s0)

        # Compressibility
        Wcompress = self.W_compress(J)

        # Active stress
        Wactive = self.Wactive(I4f, diff=0)

        # Vicoelastic
        Wvisco = self.W_visco(F, F_dot)

        W1 = self.W_1(I1, diff=0)
        W4f = self.W_4(I4f, "f", diff=0)
        W4s = self.W_4(I4s, "s", diff=0)
        W8fs = self.W_8(I8fs, diff=0)

        W = W1 + W4f + W4s + W8fs + Wcompress + Wvisco + Wactive
        return W
