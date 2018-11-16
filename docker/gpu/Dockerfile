FROM nvidia/cuda:9.0-cudnn7-devel

MAINTAINER Satoshi Terasaki <s.terasaki@idein.jp>
# Reference: OpenCV setting
#https://github.com/ildoonet/tf-pose-estimation/blob/master/docker/Dockerfile 

# setup
RUN apt-get update -yq && apt-get install -yq build-essential cmake git pkg-config wget zip && \
apt-get install -yq libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev && \
apt-get install -yq libavcodec-dev libavformat-dev libswscale-dev libv4l-dev && \
apt-get install -yq libgtk2.0-dev && \
apt-get install -yq libatlas-base-dev gfortran && \
apt-get install -yq libcanberra-gtk-module && \
apt-get install -yq python3 python3-dev python3-pip python3-setuptools python3-tk git && \
apt-get remove -yq python-pip python3-pip && wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py
# install opencv
RUN pip3 install numpy && \
cd ~ && git clone -b 3.4.3 https://github.com/Itseez/opencv.git && \
cd opencv && mkdir build && cd build && \
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D BUILD_opencv_python3=yes -D PYTHON_EXECUTABLE=/usr/bin/python3 .. && \
make -j8 && make install && rm -rf /root/opencv/ && \
rm -rf /tmp/*.tar.gz && \
apt-get clean && rm -rf /tmp/* /var/tmp* /var/lib/apt/lists/* && \
rm -f /etc/ssh/ssh_host_* && rm -rf /usr/share/man?? /usr/share/man/??_*
# install python dependencies
RUN pip3 install pillow matplotlib scipy
RUN pip3 install chainer==5.0.0 cupy-cuda90==5.0.0 chainercv==0.11.0 ideep4py
# Use Agg backend for matplotlib
ENV DISPLAY 0