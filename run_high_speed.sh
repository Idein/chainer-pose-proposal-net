CMDNAME=`basename $0`
BASEMODELDIR=$(pwd)

if [ $# -ne 1 ]; then
    echo "Usage: $CMDNAME path/to/model" 1>&2
    exit 1
fi

xhost +local:docker
docker run --rm \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-v $(pwd):/work \
-v $BASEMODELDIR:/models \
--device=/dev/video0:/dev/video0 \
--runtime=nvidia \
-w /work \
ppn:latest python3 high_speed.py /models/$1
xhost -local:docker
