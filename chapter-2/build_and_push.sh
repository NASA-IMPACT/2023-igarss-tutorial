%%sh
# make sure sm-docker is installed.
pip install sagemaker-studio-image-build

# Change directory to chapter-2 of tutorial
cd ~/2023-igarss-tutorial/chapter-2/

# Build the docker over code build and push to ECR
sm-docker build .
