FROM rlplayground/cuda9_cudnn7:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install Conda

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-4.6.14-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc &&\
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy


ENV PATH /opt/conda/bin:$PATH

CMD mkdir /root/code
WORKDIR /root/code
# Add codebase stub
CMD mkdir /root/code/mpnet
ADD environment.yml /root/code/cmpnet/environment.yml
RUN conda env create -f /root/code/cmpnet/environment.yml
RUN echo "source activate cmpnet" >> /root/.bashrc
ENV BASH_ENV /root/.bashrc

# For some reason quadprog only installs outside
RUN ["/bin/bash","-c","source activate cmpnet && pip install quadprog"]

# Installing bc_gym_planning_env

# Installing dependencies
RUN apt-get update && apt-get install -y \
  libsm6 \
  libxext6\
  libxrender1 \
  libfontconfig1 \
  git

RUN ["/bin/bash","-c","source activate cmpnet && pip install gtimer"]

# Copy code base
COPY bc-gym-planning-env /root/code/bc-gym-planning-env/

RUN ["/bin/bash","-c","source activate cmpnet && pip install -e /root/code/bc-gym-planning-env/."]

# Add pydubins
RUN ["git","clone","https://github.com/jacobjj/pydubins.git"]
WORKDIR /root/code/pydubins

RUN ["/bin/bash","-c","source activate cmpnet && python  setup.py build_ext"]
RUN ["/bin/bash","-c","source activate cmpnet && python  setup.py install"]

ENV PYTHONPATH=/root/code/BC:/root/code/MPnet:$PYTHONPATH
WORKDIR /root/code
CMD ["bash"]
