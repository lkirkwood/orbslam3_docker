FROM docker.io/jahaniam/orbslam3:ubuntu20_noetic_cpu
    MAINTAINER Linus Kirkwood <linuskirkwood@gmail.com>

WORKDIR /
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y xorg-dev

ENV PATH $PATH:/root/.local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv venv --python=3.9 /.habsim-venv
COPY habitat-sim/requirements.txt /requirements.txt
RUN . /.habsim-venv/bin/activate && uv pip install pip setuptools -r /requirements.txt
RUN echo 'export PYTHONPATH="/habitat-sim/src_python"' >> /root/.bashrc
RUN echo 'source /.habsim-venv/bin/activate' >> /root/.bashrc

COPY render-replica-frames.py /render-replica-frames.py

# Using evo from within uv venv doesn't work as per https://github.com/astral-sh/python-build-standalone/issues/129
RUN apt-get install -y python3-pil python3-pil.imagetk
RUN pip install evo
