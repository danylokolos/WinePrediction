# Use an official Python runtime as a parent image
FROM python:3.6

RUN apt-get update

# RUN apt-get install -y --no-install-recommends \
#         libatlas-base-dev gfortran nginx supervisor

# RUN pip3 install uwsgi

# COPY ./requirements.txt /project/requirements.txt



#FROM python:3.6
# FROM debian:jessie-slim

# MAINTAINER ecoron

# RUN echo 'deb http://ftp.de.debian.org/debian jessie main' >> /etc/apt/sources.list
# RUN echo 'deb http://security.debian.org/debian-security jessie/updates main ' >> /etc/apt/sources.list
# RUN echo 'deb http://ftp.de.debian.org/debian sid main' >> /etc/apt/sources.list

# RUN apt-get update
RUN apt-get -y install nano
# RUN apt-get -y install git
# RUN apt-get -y install wget
# RUN apt-get -y install python3.6 python3.6-distutils

# RUN wget https://bootstrap.pypa.io/get-pip.py

# RUN python3.6 get-pip.py

# RUN rm get-pip.py

# RUN cd /usr/local/bin \
#   && rm -f easy_install \
#   && rm -f pip \
#   && rm -f pydoc \
#   && rm -f python

# RUN cd /usr/local/bin \
#   && ln -s easy_install-3.6 easy_install \
#   && ln -s pip3.6 pip \
#   && ln -s /usr/bin/pydoc3.6 pydoc \
#   && ln -s /usr/bin/python3.6 python

# RUN apt-get autoremove
# RUN apt-get autoclean

# RUN echo 'alias python=python3.6' >> ~/.bashrc

# RUN pip install numpy
# RUN pip install scipy
# RUN pip install scikit-learn==0.20.1
# RUN pip install lxml
# RUN pip install Flask


# CMD set PYTHONIOENCODING=utf-8

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World
ENV ISPROD yes
# Run app.py when the container launches
CMD ["python", "app.py"]