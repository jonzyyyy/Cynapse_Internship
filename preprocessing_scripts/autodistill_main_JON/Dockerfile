# Use the official PyTorch image from NVIDIA as the base image
FROM nvcr.io/nvidia/pytorch:25.03-py3
# FROM cr.cynapse.ai/cv/pytorch:gcp

# Set the working directory in the container
WORKDIR /app

# Install pip and upgrade it
RUN python -m pip install --upgrade pip

# Install system dependencies required for the installation
#RUN apt-get update && apt-get install -y \
#    git \
#    && rm -rf /var/lib/apt/lists/*

RUN pip install python-dotenv

# Install Autodistill from source
COPY autodistill /app/autodistill
WORKDIR /app/autodistill
RUN pip install -e .

# Install Autodistill Grounding Dino
COPY autodistill-grounding-dino /app/autodistill-grounding-dino
WORKDIR /app/autodistill-grounding-dino
RUN pip install -e .

# Install Autodistill YOLOv11
COPY autodistill-yolov11 /app/autodistill-yolov11
WORKDIR /app/autodistill-yolov11
RUN pip install -e .