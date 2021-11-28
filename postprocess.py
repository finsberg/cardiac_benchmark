from pathlib import Path

import dolfin
import h5py
import numpy as np
from dolfin import FiniteElement  # noqa: F401
from dolfin import tetrahedron  # noqa: F401
from dolfin import VectorElement  # noqa: F401


class DataCollector:
    def __init__(self, path, u) -> None:
        self._path = Path(path)
        if self._path.is_file():
            # Delete file if is allready exist
            self._path.unlink()
        self._path = path
        self.u = u
        mesh = u.function_space().mesh()
        self._comm = mesh.mpi_comm()

        # Save mesh
        with dolfin.HDF5File(self._comm, self.path, "w") as h5file:
            h5file.write(mesh, "/mesh")

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
    def __init__(self, path) -> None:
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

        self.mesh = dolfin.Mesh()

        self._h5file = dolfin.HDF5File(
            self.mesh.mpi_comm(),
            self.path,
            "r",
        )
        self._h5file.read(self.mesh, "/mesh", False)

        V = dolfin.FunctionSpace(self.mesh, self.signature)
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
        xdmf = dolfin.XDMFFile(self.mesh.mpi_comm(), path)
        for t in self.time_stamps:
            u = self.get(t)
            xdmf.write(u, t)
        xdmf.close()

    def deformation_at_point(self, p):
        us = []
        for t in self.time_stamps:
            u = self.get(t)
            us.append(u(p))
        return np.array(us)
