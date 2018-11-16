xhost +local:docker
docker run --rm \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-v $PWD:/work \
-w /work \
--device=/dev/video0:/dev/video0 \
--runtime=nvidia \
ppn:latest python3 video.py
xhost -local:docker