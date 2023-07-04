from pathlib import Path

from .geometry import BiVGeometry


def run(mesh_file: Path, fiber_file: Path, sheet_file: Path, sheet_normal_file: Path):
    geo = BiVGeometry(
        mesh_file=mesh_file,
        fiber_file=fiber_file,
        sheet_file=sheet_file,
        sheet_normal_file=sheet_normal_file,
    )

    print(geo)
