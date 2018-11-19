from scipy.io import loadmat
import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('matfile')
    parser.add_argument('output')
    args = parser.parse_args()

    mat = loadmat(args.matfile)

    annotations = []
    joint_map = {
        0: "r_ankle",
        1: "r_knee",
        2: "r_hip",
        3: "l_hip",
        4: "l_knee",
        5: "l_ankle",
        6: "pelvis",
        7: "thorax",
        8: "upper_neck",
        9: "head_top",
        10: "r_wrist",
        11: "r_elbow",
        12: "r_shoulder",
        13: "l_shoulder",
        14: "l_elbow",
        15: "l_wrist"
    }

    for i, (anno, train_flag) in enumerate(
        zip(mat['RELEASE']['annolist'][0, 0][0],
            mat['RELEASE']['img_train'][0, 0][0])):

        img_fn = anno['image']['name'][0, 0][0]
        train_flag = int(train_flag)

        head_rect = []
        if 'x1' in str(anno['annorect'].dtype):
            head_rect = zip(
                [x1[0, 0] for x1 in anno['annorect']['x1'][0]],
                [y1[0, 0] for y1 in anno['annorect']['y1'][0]],
                [x2[0, 0] for x2 in anno['annorect']['x2'][0]],
                [y2[0, 0] for y2 in anno['annorect']['y2'][0]])

        if 'annopoints' in str(anno['annorect'].dtype):
            annopoints = anno['annorect']['annopoints'][0]
            head_x1s = anno['annorect']['x1'][0]
            head_y1s = anno['annorect']['y1'][0]
            head_x2s = anno['annorect']['x2'][0]
            head_y2s = anno['annorect']['y2'][0]
            for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(
                    annopoints, head_x1s, head_y1s, head_x2s, head_y2s):
                if len(annopoint) == 0:
                    continue
                else:
                    head_rect = [float(head_x1[0, 0]),
                                 float(head_y1[0, 0]),
                                 float(head_x2[0, 0]),
                                 float(head_y2[0, 0])]

                    # joint coordinates
                    annopoint = annopoint['point'][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                    x = [x[0, 0] for x in annopoint['x'][0]]
                    y = [y[0, 0] for y in annopoint['y'][0]]
                    joint_pos = {}
                    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                        #joint_pos[str(_j_id)] = [float(_x), float(_y)]
                        joint_pos[joint_map[int(_j_id)]] = [float(_x), float(_y)]

                    # visiblity list
                    if 'is_visible' in str(annopoint.dtype):
                        vis = [v[0] if v else [0]
                               for v in annopoint['is_visible'][0]]
                        # vis = dict([(k, int(v[0])) if len(v) > 0 else v
                        vis = dict([(joint_map[int(k)], int(v[0])) if len(v) > 0 else v
                                    for k, v in zip(j_id, vis)])
                    else:
                        vis = None

                    if len(joint_pos) == 16:
                        data = {
                            'filename': img_fn,
                            'train': train_flag,
                            'head_rect': head_rect,
                            'is_visible': vis,
                            'joint_pos': joint_pos
                        }

                        annotations.append(data)
    json.dump(annotations, open(args.output, 'w'))


if __name__ == '__main__':
    main()
