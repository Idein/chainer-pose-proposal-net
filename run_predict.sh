CMDNAME=`basename $0`
BASEMODELDIR=$(pwd)

if [ $# -ne 1 ]; then
    echo "Usage: $CMDNAME path/to/model" 1>&2
    exit 1
fi

docker run --rm \
-v $(pwd):/work \
-v $BASEMODELDIR:/models \
-v ~/work/dataset/mpii_dataset:/mpii_dataset \
-v ~/work/dataset/coco_dataset:/coco_dataset \
-w /work \
idein/chainer:5.1.0 python3 predict.py /models/$1