import typing

import dolfin
import ufl

from geometry import EllipsoidGeometry
from material import HolzapfelOgden
from solver import NonlinearProblem
from solver import NonlinearSolver


def interpolate(x0, x1, alpha: float):
    r"""Interpolate beteween :math:`x_0` and :math:`x_1`
    to find `math:`x_{1-\alpha}`

    Parameters
    ----------
    x0 : T
        First point
    x1 : T
        Second point
    alpha : float
        Amount of interpolate

    Returns
    -------
    T
        `math:`x_{1-\alpha}`
    """
    return alpha * x0 + (1 - alpha) * x1


class Problem:
    """
    Class for the mehcanics problem

    For time integration we employ the generalized :math:`alpha`-method

    .. rubric:: Reference

        Silvano Erlicher, Luca Bonaventura, Oreste Bursi.
        The analysis of the Generalized-alpha method for
        non-linear dynamic problems. Computational Mechanics,
        Springer Verlag, 2002, 28, pp.83-104, doi:10.1007/s00466-001-0273-z

    """

    def __init__(
        self,
        geometry: EllipsoidGeometry,
        material: HolzapfelOgden,
        parameters: typing.Optional[typing.Dict[str, dolfin.Constant]] = None,
        function_space: str = "P_1",
        solver_parameters=None,
    ) -> None:
        """Constructor

        Parameters
        ----------
        geometry : EllipsoidGeometry
            The geometry
        material : HolzapfelOgden
            The material
        parameters : typing.Dict[str, dolfin.Constant], optional
            Problem parameters, by default None. See
            `Problem.default_parameters`
        function_space : str, optional
            A string of the form `"{family}_{degree}` representing
            the function space for the displacement, by default "P_2"
        """
        self.geometry = geometry
        self.material = material

        parameters = parameters or {}
        self.parameters = Problem.default_parameters()
        self.parameters.update(parameters)
        self._function_space = function_space

        self.solver_parameters = NonlinearSolver.default_solver_parameters()
        if solver_parameters is not None:
            self.solver_parameters.update(**solver_parameters)
        self._init_spaces()
        self._init_forms()

    def _init_spaces(self):
        """Initialize function spaces"""
        mesh = self.geometry.mesh

        family, degree = self._function_space.split("_")

        element = dolfin.VectorElement(family, mesh.ufl_cell(), int(degree))
        self.u_space = dolfin.FunctionSpace(mesh, element)
        self.u = dolfin.Function(self.u_space)
        self.u_test = dolfin.TestFunction(self.u_space)

        self.u_old = dolfin.Function(self.u_space)
        self.v_old = dolfin.Function(self.u_space)
        self.a_old = dolfin.Function(self.u_space)

    def _acceleration_form(self, a, w):
        return dolfin.inner(self.parameters["rho"] * a, w) * dolfin.dx

    def _form(self, u, v, w):
        F = dolfin.variable(dolfin.grad(u) + dolfin.Identity(3))
        J = dolfin.det(F)
        F_dot = dolfin.grad(v)
        E_dot = dolfin.variable(0.5 * (F.T * F_dot + F_dot.T * F))
        n = ufl.cofac(F) * self.N

        return (
            -dolfin.inner(self.parameters["p"] * J * n, w) * self.ds(self.endo)
            + (
                dolfin.inner(
                    dolfin.dot(self.parameters["alpha_epi"] * u, self.N)
                    + dolfin.dot(self.parameters["beta_epi"] * v, self.N),
                    dolfin.dot(w, self.N),
                )
            )
            * self.ds(self.epi)
            + (
                dolfin.inner(
                    self.parameters["alpha_top"] * u + self.parameters["beta_top"] * v,
                    w,
                )
            )
            * self.ds(self.top)
            + dolfin.inner(
                dolfin.diff(self.material.strain_energy(F), F),
                dolfin.grad(w),
            )
            * dolfin.dx
            + dolfin.inner(
                F * dolfin.diff(self.material.W_visco(E_dot), E_dot),
                F.T * dolfin.grad(w),
            )
            * dolfin.dx
        )

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
        dt2 = dt**2
        beta = self._beta
        return (u - (u_old + dt * v_old + (0.5 - beta) * dt2 * a_old)) / (beta * dt2)

    def _update_fields(self) -> None:
        """Update old values of displacement, velocity
        and accelaration
        """
        self.v_old.vector()[:] = self.v(as_vector=True)
        self.a_old.vector()[:] = self.a(as_vector=True)
        self.u_old.vector()[:] = self.u.vector()

    @property
    def ds(self):
        return dolfin.ds(domain=self.geometry.mesh, subdomain_data=self.geometry.ffun)

    @property
    def endo(self):
        return self.geometry.markers["ENDO"][0]

    @property
    def epi(self):
        return self.geometry.markers["EPI"][0]

    @property
    def top(self):
        return self.geometry.markers["BASE"][0]

    @property
    def N(self):
        return dolfin.FacetNormal(self.geometry.mesh)

    def _init_forms(self) -> None:
        """Initialize ufl forms"""
        w = self.u_test

        # Markers
        if self.geometry.markers is None:
            raise RuntimeError("Missing markers in geometry")

        alpha_m = self.parameters["alpha_m"]
        alpha_f = self.parameters["alpha_f"]

        self._virtual_work = self._acceleration_form(
            interpolate(self.a_old, self.a(), alpha_m),
            w,
        ) + self._form(
            interpolate(self.u_old, self.u, alpha_f),
            interpolate(self.v_old, self.v(), alpha_f),
            w,
        )
        self._jacobian = dolfin.derivative(
            self._virtual_work,
            self.u,
            dolfin.TrialFunction(self.u_space),
        )

        # breakpoint()

        # bcs = dolfin.DirichletBC(self.u_space, dolfin.Constant((0.0, 0.0, 0.0), )

        self._problem = NonlinearProblem(
            J=self._jacobian,
            F=self._virtual_work,
            bcs=[],
        )
        self.solver = NonlinearSolver(
            self._problem,
            self.u,
            parameters=self.solver_parameters,
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
        """Parameter in the generalized alpha-method"""
        return dolfin.Constant(
            0.5 + self.parameters["alpha_f"] - self.parameters["alpha_m"],
        )

    @property
    def _beta(self) -> dolfin.Constant:
        """Parameter in the generalized alpha-method"""
        return dolfin.Constant((self._gamma + 0.5) ** 2 / 4.0)

    def solve(self) -> bool:
        """Solve the system"""
        _, conv = self.solver.solve()

        if not conv:
            self.u.assign(self.u_old)
            self._init_forms()
            return False

        self._update_fields()
        return True
