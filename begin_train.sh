cat config.ini
docker run --rm \
-v $(pwd):/work \
-v ~/work/dataset/mpii_dataset:/mpii_dataset \
-v ~/work/dataset/coco_dataset:/coco_dataset \
--runtime=nvidia \
--name ppn_idein \
-w /work \
idein/chainer:5.1.0 \
python3 train.py
