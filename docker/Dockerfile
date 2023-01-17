FROM finsberg/fenics-gmsh

# We upgrade pip and setuptools
RUN python3 -m pip install pip setuptools --upgrade

RUN python -m pip install scipy matplotlib meshio h5py --no-binary=h5py

# We set the working directory to install docker dependencies
WORKDIR /tmp/

# Install the Python-module
RUN git clone --branch ${REPO_BRANCH} --single-branch https://github.com/finsberg/cardiac_benchmark
RUN python3 -m pip install  --no-cache-dir ./cardiac_benchmark


# We remove the contents of the temporary directory to minimize the size of the image
RUN rm -rf /tmp
