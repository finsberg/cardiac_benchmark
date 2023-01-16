FROM finsberg/fenics-gmsh

RUN python -m pip install scipy matplotlib meshio h5py --no-binary=h5py
