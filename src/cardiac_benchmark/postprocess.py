from pathlib import Path
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import dolfin
import h5py
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfin import FiniteElement  # noqa: F401
from dolfin import tetrahedron  # noqa: F401
from dolfin import VectorElement  # noqa: F401

from .geometry import load_geometry
from .geometry import save_geometry
from .material import HolzapfelOgden
from .problem import Problem


class SavedProblem(NamedTuple):
    problem: Problem
    time_stamps_str: np.ndarray
    time_stamps: np.ndarray
    signature: ufl.finiteelement.FiniteElementBase


def save_problem(
    fname,
    problem: Problem,
    pressure_parameters: Dict[str, float],
):
    path = Path(fname)
    if path.is_file():
        path.unlink()
    save_geometry(fname, problem.geometry)

    with h5py.File(fname, "a") as h5file:
        group = h5file.create_group("problem_parameters")
        for k, v in problem.parameters.items():
            group.create_dataset(k, data=float(v))

        group = h5file.create_group("material_parameters")
        for k, v in problem.material.parameters.items():
            group.create_dataset(k, data=float(v))

        group = h5file.create_group("pressure_parameters")
        for k, v in pressure_parameters.items():
            group.create_dataset(k, data=float(v))

        h5file.create_group("p")
        h5file.create_group("tau")


def load_problem(fname) -> SavedProblem:
    path = Path(fname)
    if not path.is_file():
        raise FileNotFoundError(f"File {path} does not exist")

    required_keys = ["mesh", "u", "problem_parameters", "material_parameters"]
    with h5py.File(path, "r") as h5file:

        # Check that we have the required keys
        for key in required_keys:
            if key not in h5file:
                raise ValueError(f"Missing {key} in results file. Cannot load data")

        time_stamps_str = sorted(h5file["u"].keys(), key=lambda x: float(x))
        time_stamps = np.array(time_stamps_str, dtype=float)

        if len(time_stamps_str) == 0:
            raise ValueError(
                "No timestamps found in results file. Cannot load data",
            )

        signature = h5file["u"][time_stamps_str[0]].attrs["signature"]
        signature = eval(signature)

        material_parameters = {}
        for k, v in h5file["material_parameters"].items():
            material_parameters[k] = v[...].tolist()

        problem_parameters = {}
        for k, v in h5file["problem_parameters"].items():
            problem_parameters[k] = v[...].tolist()

    geometry = load_geometry(path)

    tau = dolfin.Constant(0.0)
    material = HolzapfelOgden(
        f0=geometry.f0,
        n0=geometry.n0,
        tau=tau,
        parameters=material_parameters,
    )

    problem = Problem(
        geometry=geometry,
        material=material,
        function_space=f"{signature.family()}_{signature.degree()}",
        parameters=problem_parameters,
    )

    return SavedProblem(
        problem=problem,
        time_stamps_str=time_stamps_str,
        time_stamps=time_stamps,
        signature=signature,
    )


class DataCollector:
    def __init__(
        self,
        path,
        problem: Problem,
        pressure_parameters: Dict[str, float],
    ) -> None:
        self._path = Path(path)
        if self._path.is_file():
            # Delete file if is allready exist
            self._path.unlink()
        self._path = path
        self.u = problem.u_old
        self.v = problem.v_old
        self.a = problem.a_old
        self.p = problem.parameters["p"]
        self.tau = problem.material.tau

        self._comm = dolfin.MPI.comm_world
        if problem.geometry.mesh is not None:
            self._comm = problem.geometry.mesh.mpi_comm()

        save_problem(path, problem, pressure_parameters)

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
            h5file.write(self.v, f"/v/{t:.4f}")
            h5file.write(self.a, f"/a/{t:.4f}")

        with h5py.File(self.path, "a") as h5file:
            h5file["p"].create_dataset(f"{t:.4f}", data=float(self.p))
            h5file["tau"].create_dataset(f"{t:.4f}", data=float(self.tau))


class DataLoader:
    def __init__(
        self,
        result_file: Union[str, Path],
        outdir: Optional[Union[str, Path]] = None,
    ) -> None:
        self._path = Path(result_file)
        if not self._path.is_file():
            raise FileNotFoundError(f"File {result_file} does not exist")

        self._problem = load_problem(self._path)

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

        self._h5file_py = h5py.File(self.path, "r")

        if self.geometry.markers is None:
            raise RuntimeError("Cannot load markers")
        self.ds = dolfin.Measure(
            "exterior_facet",
            domain=mesh,
            subdomain_data=self.geometry.ffun,
        )(self.geometry.markers["ENDO"][0])

        V = dolfin.FunctionSpace(mesh, self.signature)
        self.u = dolfin.Function(V, name="displacement")
        self.v = dolfin.Function(V, name="velocity")
        self.a = dolfin.Function(V, name="acceleration")

        self.stress_space = dolfin.FunctionSpace(self.geometry.mesh, "CG", 1)

    def __del__(self):
        self._h5file.close()
        self._h5file_py.close()

    @property
    def problem(self):
        return self._problem.problem

    @property
    def signature(self):
        return self._problem.signature

    @property
    def time_stamps(self):
        return self._problem.time_stamps

    @property
    def time_stamps_str(self):
        return self._problem.time_stamps_str

    @property
    def geometry(self):
        return self.problem.geometry

    def _check_t(self, t):
        if not isinstance(t, str):
            t = f"{t:.4f}"

        if t not in self.time_stamps_str:
            raise KeyError(f"Invalid time stamp {t}")

    def get_u(self, t):

        self._check_t(t)
        self._h5file.read(self.u, f"u/{t}/")
        return self.u

    def get_u_p_tau(self, t):
        self._check_t(t)
        self._h5file.read(self.u, f"u/{t}/")
        p = self._h5file_py["p"][t][...].tolist()
        tau = self._h5file_py["tau"][t][...].tolist()
        return (self.u, p, tau)

    def get(self, t):

        self._check_t(t)

        self._h5file.read(self.u, f"u/{t}/")
        self._h5file.read(self.v, f"v/{t}/")
        self._h5file.read(self.a, f"a/{t}/")
        return self.u, self.v, self.a

    @property
    def path(self) -> str:
        return self._path.as_posix()

    def von_Mises_stress_at_point(
        self,
        point: Tuple[float, float, float],
    ) -> List[float]:
        print(f"Compute von Mises stress at point {point}")
        stress = []
        for t in self.time_stamps_str:
            print(f"Time {t}", end="\r")
            stress.append(self.von_Mises_stress_at_timepoint(t)(point))
        return stress

    def von_Mises_stress_at_timepoint(self, t):
        u, p, tau = self.get_u_p_tau(t)
        self.problem.u.assign(u)
        self.problem.parameters["p"] = p
        self.problem.parameters["tau"] = tau
        return dolfin.project(self.problem.von_Mises(), self.stress_space)

    def to_xdmf(self, path):
        print("Write displacement to xdmf")
        xdmf = dolfin.XDMFFile(self.geometry.mesh.mpi_comm(), path)
        for t in self.time_stamps:
            print(f"Time {t}", end="\r")
            u, v, a = self.get(t)

            xdmf.write_checkpoint(u, "displacement", t, append=True)
            xdmf.write_checkpoint(v, "velocity", t, append=True)
            xdmf.write_checkpoint(a, "acceleration", t, append=True)
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
            u = self.get_u(t)
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
            u, v, a = self.get(t)

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

    def postprocess_all(self, folder: Optional[Union[str, Path]] = None):

        if folder is None:
            folder = self._path.with_suffix("")
        outfolder = Path(folder)
        outfolder.mkdir(parents=True, exist_ok=True)

        comm = self.geometry.mesh.mpi_comm()

        u_path = outfolder / "displacement.xdmf"
        u_xdmf = dolfin.XDMFFile(comm, u_path.as_posix())

        von_Mises_path = outfolder / "von_Mises_stress.xdmf"
        von_mises_xdmf = dolfin.XDMFFile(comm, von_Mises_path.as_posix())
        s = dolfin.Function(self.stress_space)
        p0 = (0.025, 0.03, 0)
        p1 = (0, 0.03, 0)

        vols = []
        up0 = []
        up1 = []
        von_mises_p0 = []
        von_mises_p1 = []
        for t in self.time_stamps_str:
            print(f"Time {t}", end="\r")
            u = self.get_u(t)

            s.assign(self.von_Mises_stress_at_timepoint(t))
            von_mises_p0.append(s(p0))
            von_mises_p1.append(s(p1))

            u_xdmf.write(u, float(t))
            von_mises_xdmf.write(s, float(t))

            up0.append(u(p0))
            up1.append(u(p1))
            vols.append(self._volume_at_timepoint(u))
        u_xdmf.close()
        von_mises_xdmf.close()

        plot_componentwise_displacement(
            up0=np.array(up0),
            up1=np.array(up1),
            time_stamps=self.time_stamps,
            fname=outfolder / "componentwise_displacement.png",
        )
        plot_von_Mises_stress(
            von_mises_p0,
            von_mises_p1,
            time_stamps=self.time_stamps,
            fname=outfolder / "von_Mises_stress.png",
        )
        plot_volume(
            volumes=vols,
            time_stamps=self.time_stamps,
            fname=outfolder / "volume.png",
        )


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


def plot_von_Mises_stress(
    sp0,
    sp1,
    time_stamps,
    fname="von_Mises_stress.png",
):
    fname = Path(fname).with_suffix(".png")

    basefname = fname.with_suffix("").as_posix()
    np.save(Path(basefname + "_sp0").with_suffix(".npy"), sp0)
    np.save(Path(basefname + "_sp1").with_suffix(".npy"), sp1)

    fig, ax = plt.subplots()
    ax.plot(time_stamps, sp0, label=r"$\sigma_v(p_0)$")
    ax.plot(time_stamps, sp1, label=r"$\sigma_v(p_1)$")
    ax.set_ylabel("von Mises stress [Pa]")
    ax.set_xlabel("Time [s]")
    ax.legend()
    ax.grid()
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


def plot_activation_pressure_function(t, act, pressure, outdir):

    fig, ax = plt.subplots()
    ax.plot(t, act)
    ax.set_title("Activation fuction \u03C4(t)")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_xlabel("Time [s]")
    fig.savefig(Path(outdir) / "activation_function.png")

    fig, ax = plt.subplots()
    ax.plot(t, pressure)
    ax.set_title("Pressure fuction p(t)")
    ax.set_ylabel("Pressure [Pa]")
    ax.set_xlabel("Time [s]")
    fig.savefig(Path(outdir) / "pressure_function.png")


if __name__ == "__main__":
    loader = DataLoader("results.h5")
    loader.postprocess_all()
    loader.compare_results()
