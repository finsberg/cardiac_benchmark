from pathlib import Path

import dolfin

from geometry import EllipsoidGeometry
from material import HolzapfelOgden
from problem import Problem


dolfin.parameters["form_compiler"]["quadrature_degree"] = 4
# TODO: Should we add more compiler flags?


def main():
    path = Path("geometry.h5")
    if not path.is_file():
        geo = EllipsoidGeometry.from_parameters()
        geo.save(path)
    geo = EllipsoidGeometry.from_file(path)

    material = HolzapfelOgden(f0=geo.f0, s0=geo.s0)

    problem = Problem(geometry=geo, material=material)
    problem.solve()


if __name__ == "__main__":
    main()
