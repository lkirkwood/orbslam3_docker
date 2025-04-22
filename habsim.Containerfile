FROM docker.io/jahaniam/orbslam3:ubuntu20_noetic_cpu
    MAINTAINER Linus Kirkwood <linuskirkwood@gmail.com>

WORKDIR /
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y xorg-dev

ENV PATH $PATH:/root/.local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv venv --python=3.9 /.habsim-venv
COPY habitat-sim/requirements.txt /requirements.txt
RUN . /.habsim-venv/bin/activate && uv pip install pip setuptools -r /requirements.txt

# RUN . .venv/bin/activate && \
#     export CXXFLAGS="-I$(uv python dir)/$(ls $(uv python dir))/include/python3.9" && \
#     uv pip install setuptools pip && ./build.sh --headless -j4

# FROM docker.io/jahaniam/orbslam3:ubuntu20_noetic_cpu

# COPY --from=habsim /habitat-sim /habitat-sim

# RUN mkdir -p ~/miniconda3
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# RUN chmod +x ~/miniconda3/miniconda.sh
# RUN ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# RUN rm ~/miniconda3/miniconda.sh
# RUN . ~/miniconda3/bin/activate && conda init --all

# RUN . ~/miniconda3/bin/activate && conda create -n habitat python=3.9 cmake=3.14.0 && \
#     conda activate habitat && conda install habitat-sim headless -c conda-forge -c aihabitat

# FROM docker.io/jahaniam/orbslam3:ubuntu20_noetic_cpu
#     MAINTAINER Linus Kirkwood <linuskirkwood@gmail.com>

# RUN apt-get update && apt-get install -y git-lfs
# RUN git lfs install
