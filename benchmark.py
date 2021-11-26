from pathlib import Path
from geometry import EllipsoidGeometry


def main():
    path = Path("geometry.h5")
    if not path.is_file():
        geo = EllipsoidGeometry.from_parameters()
        geo.save(path)
    geo = EllipsoidGeometry.from_file(path)


if __name__ == "__main__":
    main()
