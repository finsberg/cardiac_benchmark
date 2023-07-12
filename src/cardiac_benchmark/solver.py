"""Copied from pulse"""
import logging
import time
from typing import Tuple

import dolfin


logger = logging.getLogger(__name__)


class NonlinearProblem(dolfin.NonlinearProblem):
    """NonlinearProblem that re-uses the assembly matrix"""

    def __init__(self, J, F, bcs, **kwargs):
        super().__init__(**kwargs)
        self._J = J
        self._F = F
        self.bcs = bcs

    def F(self, b: dolfin.PETScVector, x: dolfin.PETScVector):
        dolfin.assemble(self._F, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A: dolfin.PETScMatrix, x: dolfin.PETScVector):
        dolfin.assemble(self._J, tensor=A)
        for bc in self.bcs:
            bc.apply(A)


class NonlinearSolver:
    def __init__(
        self,
        problem: NonlinearProblem,
        state,
        parameters=None,
    ):
        dolfin.PETScOptions.clear()
        self.update_parameters(parameters)
        self._problem = problem
        self._state = state

        self._solver = dolfin.PETScSNESSolver(state.function_space().mesh().mpi_comm())
        self._solver.set_from_options()

        self._solver.parameters.update(self.parameters)
        self._snes = self._solver.snes()
        self._snes.setConvergenceHistory()

        logger.info(f"Linear Solver : {self._solver.parameters['linear_solver']}")
        logger.info(f"Preconditioner:  {self._solver.parameters['preconditioner']}")
        logger.info(f"atol: {self._solver.parameters['absolute_tolerance']}")
        logger.info(f"rtol: {self._solver.parameters['relative_tolerance']}")
        logger.info(f" Size          : {self._state.function_space().dim()}")
        dolfin.PETScOptions.clear()

    def update_parameters(self, parameters):
        """Update solver parameters"""
        ps = NonlinearSolver.default_solver_parameters()
        if hasattr(self, "parameters"):
            ps.update(self.parameters)
        if parameters is not None:
            ps.update(parameters)
        petsc = ps.pop("petsc")

        for k, v in petsc.items():
            if v is not None:
                dolfin.PETScOptions.set(k, v)
        self.verbose = ps.pop("verbose", False)
        if self.verbose:
            dolfin.PETScOptions.set("ksp_monitor")
            dolfin.PETScOptions.set("log_view")
            dolfin.PETScOptions.set("ksp_view")
            dolfin.PETScOptions.set("pc_view")
            dolfin.PETScOptions.set("snes_monitor_lg_residualnorm")
            dolfin.PETScOptions.set("mat_superlu_dist_statprint", True)
            ps["lu_solver"]["report"] = True
            ps["lu_solver"]["verbose"] = True
            ps["report"] = True
            ps["krylov_solver"]["monitor_convergence"] = True
        self.parameters = ps

    @staticmethod
    def default_solver_parameters():
        """Default solver parameters"""
        linear_solver = "superlu_dist"
        return {
            "petsc": {
                "ksp_type": "preonly",
                "ksp_norm_type": "preconditioned",
                "pc_type": "lu",
                "mat_mumps_icntl_33": 0,
            },
            "verbose": False,
            "linear_solver": linear_solver,
            "preconditioner": "lu",
            "error_on_nonconvergence": False,
            "relative_tolerance": 1e-5,
            "absolute_tolerance": 1e-5,
            "maximum_iterations": 50,
            "report": False,
            "krylov_solver": {
                "absolute_tolerance": 1e-10,
                "relative_tolerance": 1e-10,
                "maximum_iterations": 1000,
                "monitor_convergence": False,
            },
            "lu_solver": {"report": False, "symmetric": False, "verbose": False},
        }

    def solve(self) -> Tuple[int, bool]:
        """Solves the problem

        Returns
        -------
        Tuple[int, bool]
            (Number of iterations, Converged)
        """

        logger.info("Solving NonLinearProblem...")

        start = time.perf_counter()
        self._solver.solve(self._problem, self._state.vector())
        end = time.perf_counter()

        msg = f"\nDone in {end - start:.3f} s"

        residuals = self._snes.getConvergenceHistory()[0]
        num_iterations = self._snes.getLinearSolveIterations()
        msg += f"\nIterations:   {num_iterations}"
        if num_iterations > 0:
            msg += f"\nResidual      : {residuals[-1]}"

        logger.info(msg)
        return num_iterations, self._snes.converged
