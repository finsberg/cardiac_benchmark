from pathlib import Path

import dolfin
import h5py
import matplotlib.pyplot as plt
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

    def deformation_at_point(self, p):
        print(f"Compute deformation at point {p}")
        us = []
        for t in self.time_stamps:
            print(f"Deformation at time {t}", end="\r")
            us.append(self._deformation_at_time_point(self.get(t)(p)))

        return np.array(us)

    def _volume_form(self, u):
        X = dolfin.SpatialCoordinate(self.geometry.mesh) - self._shift
        N = dolfin.FacetNormal(self.geometry.mesh)
        F = dolfin.grad(u) + dolfin.Identity(3)

        u1 = ufl.as_vector([0.0, u[1], 0.0])
        X1 = ufl.as_vector([0.0, X[1], 0.0])
        return (-1.0 / 1.0) * dolfin.dot(X1 + u1, ufl.cofac(F) * N)

    def _volume_at_timepoint(self, u):
        return dolfin.assemble(self._volume_form(u) * self.ds)

    def cavity_volume(self):
        print("Compute cavity volume...")
        vols = []
        for t in self.time_stamps:
            print(f"Volume at time {t}", end="\r")
            u = self.get(t)
            vols.append(self._volume_at_timepoint(u))
        return np.array(vols)

    def compare_results(
        self,
        disp_path="cardiac_benchmark/2021_10_29/results/displacement_vs_time.npz",
        vol_path="cardiac_benchmark/2021_10_29/results/volume_vs_time.npz",
    ):
        """Compare results with provided results for
        compononentwise displacement and volumes

        Parameters
        ----------
        disp_path : str
            Path to file containing results for
            component-wise displacements
        vol_path : str
            Path to file containing results for
            volume

        Raises
        ------
        FileNotFoundError
            If any of the files are not found
        """

        # Displacements
        disp_path = Path(disp_path)
        if not disp_path.is_file():
            raise FileNotFoundError(f"Cannot find file {disp_path}")
        results_disp = np.load(disp_path, allow_pickle=True)
        time_stamps_cmp = results_disp["times.npy"]
        results_disp_dict = results_disp["disp_dict.npy"].item()
        up0_cmp = np.array(list(results_disp_dict["top"].values())).T
        up1_cmp = np.array(list(results_disp_dict["middle"].values())).T

        # Volumes
        vol_path = Path(vol_path)
        if not vol_path.is_file():
            raise FileNotFoundError(f"Cannot find file {vol_path}")
        volumes_cmp = np.load(vol_path, allow_pickle=True)["vol_lst.npy"]

        p0 = (0.025, 0.03, 0)
        p1 = (0, 0.03, 0)

        vols = []
        up0 = []
        up1 = []
        for t in self.time_stamps:
            print(f"Time {t}", end="\r")
            u = self.get(t)
            up0.append(u(p0))
            up1.append(u(p1))
            vols.append(self._volume_at_timepoint(u))

        plot_componentwise_displacement_comparison(
            up0=np.array(up0),
            up1=np.array(up1),
            time_stamps=self.time_stamps,
            up0_cmp=up0_cmp,
            up1_cmp=up1_cmp,
            time_stamps_cmp=time_stamps_cmp,
        )
        plot_volume_comparison(
            volumes=vols,
            time_stamps=self.time_stamps,
            volumes_cmp=volumes_cmp,
            time_stamps_cmp=time_stamps_cmp,
        )

    def postprocess_all(self):
        xdmf = dolfin.XDMFFile(self.geometry.mesh.mpi_comm(), "u.xdmf")
        p0 = (0.025, 0.03, 0)
        p1 = (0, 0.03, 0)

        # p0 = (0.02647058823529411, -1.463099651233471e-18, -0.02389423060187047)
        # p1 = (0.02647058823529411, 4.389298953700411e-18, 0.02389423060187047)
        # p2 = (0.02647058823529411, 0, 0)
        # p3 = (0.02647058823529411, 0.02389423060187047, -2.926199302466941e-18)

        vols = []
        up0 = []
        up1 = []
        for t in self.time_stamps:
            print(f"Time {t}", end="\r")
            u = self.get(t)
            xdmf.write(u, t)
            up0.append(u(p0))
            up1.append(u(p1))
            vols.append(self._volume_at_timepoint(u))
        xdmf.close()

        plot_componentwise_displacement(
            up0=np.array(up0),
            up1=np.array(up1),
            time_stamps=self.time_stamps,
        )

        plot_volume(volumes=vols, time_stamps=self.time_stamps)


def plot_componentwise_displacement(
    up0,
    up1,
    time_stamps,
    fname="componentwise_displacement.png",
):
    fname = Path(fname).with_suffix(".png")

    basefname = fname.with_suffix("").as_posix()
    np.save(Path(basefname + "_up0").with_suffix(".npy"), up0)
    np.save(Path(basefname + "_up1").with_suffix(".npy"), up1)

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(time_stamps, up0[:, 0], label="x")
    ax[0].plot(time_stamps, up0[:, 1], label="y")
    ax[0].plot(time_stamps, up0[:, 2], label="z")
    ax[0].set_ylabel("$u(p_0)$[m]")

    ax[1].plot(time_stamps, up1[:, 0], label="x")
    ax[1].plot(time_stamps, up1[:, 1], label="y")
    ax[1].plot(time_stamps, up1[:, 2], label="z")
    ax[1].set_ylabel("$u(p_1)$[m]")
    ax[1].set_xlabel("Time [s]")

    for axi in ax:
        axi.legend()
        axi.grid()
    fig.savefig(fname, dpi=300)


def plot_componentwise_displacement_comparison(
    up0,
    up1,
    time_stamps,
    up0_cmp,
    up1_cmp,
    time_stamps_cmp,
    fname="componentwise_displacement_comparison.png",
):
    fname = Path(fname).with_suffix(".png")

    basefname = fname.with_suffix("").as_posix()
    np.save(Path(basefname + "_up0").with_suffix(".npy"), up0)
    np.save(Path(basefname + "_up1").with_suffix(".npy"), up1)

    fig, ax = plt.subplots(2, 2, sharex=True)
    ax[0, 0].plot(time_stamps, up0[:, 0], label="x")
    ax[0, 0].plot(time_stamps, up0[:, 1], label="y")
    ax[0, 0].plot(time_stamps, up0[:, 2], label="z")
    ax[0, 0].plot(time_stamps_cmp, up0_cmp[:, 0], linestyle="--", label="x (cmp)")
    ax[0, 0].plot(time_stamps_cmp, up0_cmp[:, 1], linestyle="--", label="y (cmp)")
    ax[0, 0].plot(time_stamps_cmp, up0_cmp[:, 2], linestyle="--", label="z (cmp)")
    ax[0, 0].set_title("$u(p_0)$[m]")

    ax[1, 0].plot(
        time_stamps,
        up0[:, 0] - np.interp(time_stamps, time_stamps_cmp, up0_cmp[:, 0]),
        label="x - x (cmp)",
    )
    ax[1, 0].plot(
        time_stamps,
        up0[:, 1] - np.interp(time_stamps, time_stamps_cmp, up0_cmp[:, 1]),
        label="y - y (cmp)",
    )
    ax[1, 0].plot(
        time_stamps,
        up0[:, 2] - np.interp(time_stamps, time_stamps_cmp, up0_cmp[:, 2]),
        label="z - z (cmp)",
    )
    ax[1, 0].set_xlabel("Time [s]")

    ax[0, 1].plot(time_stamps, up1[:, 0], label="x")
    ax[0, 1].plot(time_stamps, up1[:, 1], label="y")
    ax[0, 1].plot(time_stamps, up1[:, 2], label="z")
    ax[0, 1].plot(time_stamps_cmp, up1_cmp[:, 0], linestyle="--", label="x (cmp)")
    ax[0, 1].plot(time_stamps_cmp, up1_cmp[:, 1], linestyle="--", label="y (cmp)")
    ax[0, 1].plot(time_stamps_cmp, up1_cmp[:, 2], linestyle="--", label="z (cmp)")
    ax[0, 1].set_title("$u(p_1)$[m]")

    ax[1, 1].plot(
        time_stamps,
        up1[:, 0] - np.interp(time_stamps, time_stamps_cmp, up1_cmp[:, 0]),
        label="x - x (cmp)",
    )
    ax[1, 1].plot(
        time_stamps,
        up1[:, 1] - np.interp(time_stamps, time_stamps_cmp, up1_cmp[:, 1]),
        label="y - y (cmp)",
    )
    ax[1, 1].plot(
        time_stamps,
        up1[:, 2] - np.interp(time_stamps, time_stamps_cmp, up1_cmp[:, 2]),
        label="z - z (cmp)",
    )

    for axi in ax.flatten():
        axi.legend()
        axi.grid()
    fig.tight_layout()
    fig.savefig(fname, dpi=300)


def plot_volume(volumes, time_stamps, fname="volume.png"):
    fname = Path(fname).with_suffix(".png")
    basefname = fname.with_suffix("").as_posix()
    np.save(Path(basefname + "_volumes").with_suffix(".npy"), volumes)

    fig, ax = plt.subplots()
    ax.plot(time_stamps, volumes, label="volume")
    ax.set_ylabel("Volume [m^3]")
    ax.set_xlabel("Time [s]")
    ax.grid()
    ax.legend()
    ax.set_title("Volume throug time")
    fig.savefig(fname, dpi=300)


def plot_volume_comparison(
    volumes,
    time_stamps,
    volumes_cmp=None,
    time_stamps_cmp=None,
    fname="volume_comparison.png",
):
    fname = Path(fname).with_suffix(".png")
    basefname = fname.with_suffix("").as_posix()
    np.save(Path(basefname + "_volumes").with_suffix(".npy"), volumes)

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(time_stamps, volumes, label="volume")
    ax[0].plot(time_stamps_cmp, volumes_cmp, label="volume (cmp)")
    ax[1].plot(
        time_stamps,
        volumes - np.interp(time_stamps, time_stamps_cmp, volumes_cmp),
        label="volume - volume (cmp)",
    )
    ax[0].set_ylabel("Volume [m^3]")
    ax[1].set_xlabel("Time [s]")
    for axi in ax:
        axi.grid()
        axi.legend()
    ax[0].set_title("Volume throug time")
    fig.savefig(fname, dpi=300)


if __name__ == "__main__":
    loader = DataLoader("results.h5")
    loader.postprocess_all()
    loader.compare_results()
