FROM cr.cynapse.ai/cv/pytorch:yolov8
RUN apt-get update && apt-get install -y lsb-release curl gnupg2

# Add the Google Cloud packages GPG key directly using its ID
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys C0BA5CE6DC6315A3

# Download the GPG key and add it to the keyrings directory
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor --no-tty > /usr/share/keyrings/cloud.google.gpg

# Correctly set up the Google Cloud packages repository
RUN export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s) && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list

# Now install gcsfuse
RUN apt-get update && apt-get install -y gcsfuse fuse
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --batch --yes --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && apt-get install google-cloud-sdk -y

RUN pip install imagehash
