FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true


# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Python package management and basic dependencies
RUN apt-get install -y curl python3.9 python3.9-dev python3.9-distutils

# Register the version in alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.9

# Install necessary dependencies for MMS and SageMaker Inference Toolkit
RUN apt-get -y install --no-install-recommends \
    build-essential \
    ca-certificates \
    openjdk-8-jdk-headless \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && python --version \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

# Install MXNet, MMS, and SageMaker Inference Toolkit to set up MMS
RUN pip3 --no-cache-dir install openmim \
                                rasterio \
                                numpy \
                                multi-model-server \
                                sagemaker-inference \
                                retrying

RUN mim install mmcv==1.5.0 mmcv-full==1.5.0 mmsegmentation==0.30.0 mmengine==0.7.4

COPY mmsegmentation/ /opt/mmsegmentation/mmsegmentation-0.21.0-py3-none-any.whl

RUN pip3 install /opt/mmsegmentation/mmsegmentation-0.21.0-py3-none-any.whl

# Also copy over config files
# Copy entrypoint script to the image
COPY code/dockerd-entrypoint.py /usr/local/bin/dockerd-entrypoint.py
RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

RUN mkdir -p /home/model-server/

# Copy the default custom service file to handle incoming data and inference requests
COPY code/infer.py /home/model-server/infer.py

# Define an entrypoint script for the docker image
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]

# Define command to be passed to the entrypoint
CMD ["serve"]