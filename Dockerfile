FROM ubuntu:16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install Conda

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-4.6.14-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
  /opt/conda/bin/conda clean -afy

CMD mkdir /root/code
WORKDIR /root/code
# Add codebase stub
CMD mkdir /root/code/cmpnet
ADD requirements.txt /root/code/cmpnet/requirements.txt
RUN conda create -n cmpnet -f /root/code/cmpnet/requirements.txt
RUN echo "source activate cmpnet" >> /root/.bashrc
ENV BASH_ENV /root/.bashrc

CMD ["bash"]



