import concurrent.futures
import itertools as it
from pathlib import Path

import dolfin
import h5py
import numpy as np
import ufl
from dolfin import FiniteElement  # noqa: F401
from dolfin import tetrahedron  # noqa: F401
from dolfin import VectorElement  # noqa: F401

from geometry import EllipsoidGeometry
from geometry import load_geometry
from geometry import save_geometry


class DataCollector:
    def __init__(self, path, u, geometry: EllipsoidGeometry) -> None:
        self._path = Path(path)
        if self._path.is_file():
            # Delete file if is allready exist
            self._path.unlink()
        self._path = path
        self.u = u

        self._comm = dolfin.MPI.comm_world
        if geometry.mesh is not None:
            self._comm = geometry.mesh.mpi_comm()

        save_geometry(path, geometry)

    @property
    def path(self) -> str:
        return self._path.as_posix()

    def store(self, t: float) -> None:
        """Save displacement

        Parameters
        ----------
        t : float
            Time stamp
        """
        with dolfin.HDF5File(self._comm, self.path, "a") as h5file:
            h5file.write(self.u, f"/u/{t:.4f}")


class DataLoader:
    def __init__(self, path, geo) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"File {path} does not exist")

        with h5py.File(path, "r") as h5file:

            # Check that we have mesh
            if "mesh" not in h5file:
                raise ValueError("No mesh in results file. Cannot load data")

            if "u" not in h5file:
                raise ValueError("No displacement in results file. Cannot load data")

            self.time_stamps_str = sorted(h5file["u"].keys(), key=lambda x: float(x))
            self.time_stamps = np.array(self.time_stamps_str, dtype=float)

            if len(self.time_stamps_str) == 0:
                raise ValueError(
                    "No timestamps found in results file. Cannot load data",
                )

            signature = h5file["u"][self.time_stamps_str[0]].attrs["signature"]
            self.signature = eval(signature)

        self.geometry = load_geometry(path)
        if self.geometry.mesh is None:
            raise RuntimeError("Cannot load mesh")

        mesh = self.geometry.mesh
        x_max = self.geometry.mesh.coordinates().max(0)[0]
        self._shift = dolfin.Constant((x_max, 0, 0))

        self._h5file = dolfin.HDF5File(
            mesh.mpi_comm(),
            self.path,
            "r",
        )

        if self.geometry.markers is None:
            raise RuntimeError("Cannot load markers")
        self.ds = dolfin.Measure(
            "exterior_facet",
            domain=mesh,
            subdomain_data=self.geometry.ffun,
        )(self.geometry.markers["ENDO"][0])

        V = dolfin.FunctionSpace(mesh, self.signature)
        self._V_cg1 = dolfin.VectorFunctionSpace(mesh, "CG", 1)
        self.u = dolfin.Function(V)

    def get(self, t):

        if not isinstance(t, str):
            t = f"{t:.4f}"

        if t not in self.time_stamps_str:
            raise KeyError(f"Invalid time stamp {t}")

        self._h5file.read(self.u, f"u/{t}/")
        return self.u

    @property
    def path(self) -> str:
        return self._path.as_posix()

    def to_xdmf(self, path):
        print("Write displacement to xdmf")
        xdmf = dolfin.XDMFFile(self.geometry.mesh.mpi_comm(), path)
        for t in self.time_stamps:
            print(f"Time {t}", end="\r")
            u = self.get(t)
            xdmf.write(u, t)
        xdmf.close()

    def _deformation_at_time_point(self, arg):
        t, p = arg
        print(f"Deformation at time {t}", end="\r")
        return self.get(t)(p)

    def deformation_at_point(self, p):
        print(f"Compute deformation at point {p}")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            us = list(
                executor.map(
                    self._deformation_at_time_point,
                    zip(self.time_stamps, it.repeat(p)),
                ),
            )

        return np.array(us)

    def _volume_form(self, u):
        X = dolfin.SpatialCoordinate(self.geometry.mesh) - self._shift
        N = dolfin.FacetNormal(self.geometry.mesh)
        u_int = dolfin.interpolate(u, self._V_cg1)
        F = dolfin.grad(u_int) + dolfin.Identity(3)
        return (-1.0 / 3.0) * dolfin.dot(X + u, ufl.cofac(F) * N)

    def _volume_at_timepoint(self, t):
        print(f"Volume at time {t}", end="\r")
        u = self.get(t)
        return dolfin.assemble(self._volume_form(u) * self.ds)

    def cavity_volume(self):
        print("Compute cavity volume...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            vols = list(executor.map(self._volume_at_timepoint, self.time_stamps))
        return np.array(vols)
