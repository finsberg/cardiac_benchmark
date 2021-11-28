import typing

import dolfin
import ufl


class Problem:
    """


    Time integration
    ----------------

    We emplot the generalized :math:`alpha`-method

    .. rubric:: Reference

        Silvano Erlicher, Luca Bonaventura, Oreste Bursi.
        The analysis of the Generalized-alpha method for
        non-linear dynamic problems. Computational Mechanics,
        Springer Verlag, 2002, 28, pp.83-104, doi:10.1007/s00466-001-0273-z

    """

    def __init__(
        self,
        geometry,
        material,
        parameters=None,
        function_space="P_1",
    ) -> None:
        self.geometry = geometry
        self.material = material

        parameters = parameters or {}
        self.parameters = Problem.default_parameters()
        self.parameters.update(parameters)
        self._function_space = function_space
        self._init_spaces()
        self._init_forms()

    def _init_spaces(self):
        mesh = self.geometry.mesh

        family, degree = self._function_space.split("_")

        element = dolfin.VectorElement(family, mesh.ufl_cell(), int(degree))
        self.u_space = dolfin.FunctionSpace(mesh, element)
        self.u = dolfin.Function(self.u_space)
        self.u_test = dolfin.TestFunction(self.u_space)

        self.u_old = dolfin.Function(self.u_space)
        self.v_old = dolfin.Function(self.u_space)
        self.a_old = dolfin.Function(self.u_space)

    def v(
        self,
        as_vector: bool = False,
    ) -> typing.Union[dolfin.Function, dolfin.Vector]:
        r"""
        Velocity computed using the generalized
        :math:`alpha`-method

        .. math::

            v_{i+1} = v_i + (1-\gamma) \Delta t a_i + \gamma \Delta t a_{i+1}


        Parameters
        ----------
        as_vector : bool, optional
            Flag for saying whether to return the
            velocity as a function or a vector, by default False

        Returns
        -------
        typing.Union[dolfin.Function, dolfin.Vector]
            The velocity
        """
        if as_vector:
            v_old = self.v_old.vector()
            a_old = self.a_old.vector()
            a = self.a(as_vector)
        else:
            v_old = self.v_old
            a_old = self.a_old
            a = self.a()

        dt = self.parameters["dt"]
        return v_old + (1 - self._gamma) * dt * a_old + self._gamma * dt * a

    def a(
        self,
        as_vector: bool = False,
    ) -> typing.Union[dolfin.Function, dolfin.Vector]:
        r"""
        Acceleration computed using the generalized
        :math:`alpha`-method

        .. math::

            a_{i+1} = \frac{u_{i+1} - (u_i + \Delta t v_i + (0.5 - \beta) \Delta t^2 a_i)}{\beta \Delta t^2}


        Parameters
        ----------
        as_vector : bool, optional
            Flag for saying whether to return the
            acceleration as a function or a vector, by default False

        Returns
        -------
        typing.Union[dolfin.Function, dolfin.Vector]
            The acceleration
        """
        if as_vector:
            u = self.u.vector()
            u_old = self.u_old.vector()
            v_old = self.v_old.vector()
            a_old = self.a_old.vector()
        else:
            u = self.u
            u_old = self.u_old
            v_old = self.v_old
            a_old = self.a_old

        dt = self.parameters["dt"]
        dt2 = dt ** 2
        beta = self._beta
        return (u - (u_old + dt * v_old + (0.5 - beta) * dt2 * a_old)) / (beta * dt2)

    def _update_fields(self) -> None:
        """Update old values of displacement, velocity
        and accelaration
        """
        self.v_old.vector()[:] = self.v(as_vector=True)
        self.a_old.vector()[:] = self.a(as_vector=True)
        self.u_old.vector()[:] = self.u.vector()

    def _init_forms(self) -> None:
        u = self.u
        v = self.v()
        a = self.a()
        w = self.u_test

        ds = dolfin.ds(domain=self.geometry.mesh, subdomain_data=self.geometry.ffun)

        F = dolfin.variable(dolfin.grad(u) + dolfin.Identity(3))
        F_dot = dolfin.grad(v)

        # Normal vectors
        N = dolfin.FacetNormal(self.geometry.mesh)
        n = ufl.cofac(F) * N

        # Makers
        endo = self.geometry.markers["ENDO"][0]
        epi = self.geometry.markers["EPI"][0]
        top = self.geometry.markers["BASE"][0]

        internal_energy = self.material.strain_energy(F, F_dot)

        external_work = (
            dolfin.inner(self.parameters["rho"] * a, w) * dolfin.dx
            - dolfin.inner(w, self.parameters["p"] * n) * ds(endo)
            + dolfin.inner(
                (self.parameters["alpha_epi"] * u + self.parameters["beta_epi"] * v),
                w,
            )
            * ds(epi)
            + dolfin.inner(
                (self.parameters["alpha_top"] * u + self.parameters["beta_top"] * v),
                w,
            )
            * ds(top)
        )

        self._virtual_work = (
            dolfin.derivative(
                internal_energy * dolfin.dx,
                self.u,
                self.u_test,
            )
            + external_work
        )

        self._jacobian = dolfin.derivative(
            self._virtual_work,
            self.u,
            dolfin.TrialFunction(self.u_space),
        )

    @staticmethod
    def default_parameters() -> typing.Dict[str, dolfin.Constant]:
        return dict(
            alpha_top=dolfin.Constant(1e5),
            alpha_epi=dolfin.Constant(1e8),
            beta_top=dolfin.Constant(5e3),
            beta_epi=dolfin.Constant(5e3),
            p=dolfin.Constant(0.0),
            rho=dolfin.Constant(1e3),
            dt=dolfin.Constant(1e-3),
            alpha_m=dolfin.Constant(0.2),
            alpha_f=dolfin.Constant(0.4),
        )

    @property
    def _gamma(self) -> dolfin.Constant:
        return dolfin.Constant(
            0.5 + self.parameters["alpha_f"] - self.parameters["alpha_m"],
        )

    @property
    def _beta(self) -> dolfin.Constant:
        return dolfin.Constant((self._gamma + 0.5) ** 2 / 4.0)

    def solve(self) -> None:
        """The the system"""
        # FIXME: Implement a more sophisticated solver
        dolfin.solve(self._virtual_work == 0, self.u, [])
        self._update_fields()