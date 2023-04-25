FROM ghcr.io/scientificcomputing/fenics-gmsh:2023-04-21

# We set the working directory to install docker dependencies
WORKDIR /tmp/

# Install the Python-module
RUN git clone --branch ${REPO_BRANCH} --single-branch https://github.com/finsberg/cardiac_benchmark
RUN python3 -m pip install  --no-cache-dir ./cardiac_benchmark


# We remove the contents of the temporary directory to minimize the size of the image
RUN rm -rf /tmp
