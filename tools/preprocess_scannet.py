# original version from https://github.com/ardaduz/deep-video-mvs
# Copyright (c) 2020 Arda Duzceker

# Modified by Alex Rich
# From 3DVNet:
# https://github.com/alexrich021/3dvnet/blob/main/data_preprocess/preprocess_scannet.py

"""
Expects pre-extracted ScanNet using scripts provided in ScanNet python library
https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py
Creates copy with images re-sized to 640x480, ignoring images with infinite pose
Expected directory structure of src scannet:
scannet_src/
    scannetv2_train.txt
    scannetv2_val.txt
    scannetv2_test.txt
    scans/
        scene_****_**/
            scene_****_**_vh_clean_2.ply
            color/
            depth/
            intrinsic/
            pose/
    scans_test/
        scene_****_**/
            scene_****_**_vh_clean_2.ply
            color/
            depth/
            intrinsic/
            pose/
"""

import os
import torch
import numpy as np
import cv2
import json
import tqdm
import argparse
import shutil


def process_color_image(color, depth, K_color, K_depth):
    old_height, old_width = np.shape(color)[0:2]
    new_height, new_width = np.shape(depth)

    x = np.linspace(0, new_width - 1, num=new_width)
    y = np.linspace(0, new_height - 1, num=new_height)
    ones = np.ones(shape=(new_height, new_width))
    x_grid, y_grid = np.meshgrid(x, y)
    warp_grid = np.stack((x_grid, y_grid, ones), axis=-1)
    warp_grid = torch.from_numpy(warp_grid).float()
    warp_grid = warp_grid.view(-1, 3).t().unsqueeze(0)

    H = K_color.dot(np.linalg.inv(K_depth))
    H = torch.from_numpy(H).float().unsqueeze(0)

    width_normalizer = old_width / 2.0
    height_normalizer = old_height / 2.0

    warping = H.bmm(warp_grid).transpose(dim0=1, dim1=2)
    warping = warping[:, :, 0:2] / (warping[:, :, 2].unsqueeze(-1) + 1e-8)
    warping = warping.view(1, new_height, new_width, 2)
    warping[:, :, :, 0] = (warping[:, :, :, 0] - width_normalizer) / width_normalizer
    warping[:, :, :, 1] = (warping[:, :, :, 1] - height_normalizer) / height_normalizer

    image = torch.from_numpy(np.transpose(color, axes=(2, 0, 1))).float().unsqueeze(0)

    warped_image = torch.nn.functional.grid_sample(input=image,
                                                   grid=warping,
                                                   mode='nearest',
                                                   padding_mode='zeros',
                                                   align_corners=True)

    warped_image = warped_image.squeeze(0).numpy().astype(np.uint8)
    warped_image = np.transpose(warped_image, axes=(1, 2, 0))
    return warped_image


def process_scene(scene_dir_src, scene_dir_dst):
    scene_name = os.path.basename(scene_dir_src)
    data = {
        'scene': scene_name,
        'path': scene_dir_dst,
        'frames': []
    }

    if not os.path.exists(scene_dir_dst):
        os.makedirs(scene_dir_dst)
    color_dir_dst = os.path.join(scene_dir_dst, 'color')
    if not os.path.exists(color_dir_dst):
        os.makedirs(color_dir_dst)
    depth_dir_dst = os.path.join(scene_dir_dst, 'depth')
    if not os.path.exists(depth_dir_dst):
        os.makedirs(depth_dir_dst)

    # copy ground truth mesh to new folder
    gt_mesh_src = os.path.join(scene_dir_src, '{}_vh_clean_2.ply'.format(scene_name))
    gt_mesh_dst = os.path.join(scene_dir_dst, '{}_vh_clean_2.ply'.format(scene_name))
    shutil.copy(gt_mesh_src, gt_mesh_dst)
    data['gt_mesh'] = gt_mesh_dst

    K_color = np.loadtxt(os.path.join(scene_dir_src, 'intrinsic', 'intrinsic_color.txt'))[:3, :3]
    K_depth = np.loadtxt(os.path.join(scene_dir_src, 'intrinsic', 'intrinsic_depth.txt'))[:3, :3]
    data['intrinsics'] = K_depth.tolist()

    frames = sorted([f for f in os.listdir(os.path.join(scene_dir_src, 'color'))
                        if f.endswith('.jpg')], key=lambda x: int(x.split('.')[0]))

    for frame in tqdm.tqdm(frames):
        frame_id = int(frame.split('.')[0])
        fname_color_src = os.path.join(scene_dir_src, 'color', frame)
        fname_depth_src = os.path.join(scene_dir_src, 'depth', '{}.png'.format(frame_id))
        fname_color_dst = os.path.join(scene_dir_dst, 'color', '{}.jpg'.format(frame_id).zfill(9))
        fname_depth_dst = os.path.join(scene_dir_dst, 'depth', '{}.png'.format(frame_id).zfill(9))

        color = cv2.imread(fname_color_src)
        depth = cv2.imread(fname_depth_src, cv2.IMREAD_ANYDEPTH)
        P = np.loadtxt(os.path.join(scene_dir_src, 'pose', '{}.txt'.format(frame_id)))

        if not np.all(np.isfinite(P)):  # skip invalid poses
            continue

        if color.shape[:2] != depth.shape[:2]:      # avoid resizing twice
            color = process_color_image(color, depth, K_color, K_depth)
            cv2.imwrite(fname_color_dst, color)

        elif not os.path.exists(fname_color_dst):
            cv2.imwrite(fname_color_dst, color)

        if not os.path.exists(fname_depth_dst):
            cv2.imwrite(fname_depth_dst, depth)

        frame = {
            'filename_color': fname_color_dst,
            'filename_depth': fname_depth_dst,
            'pose': P.tolist()
        }
        data['frames'].append(frame)
    json.dump(data, open(os.path.join(scene_dir_dst, 'info.json'), 'w'))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str)
    parser.add_argument('--dst', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    train_txt = os.path.join(args.dst, 'scannetv2_train.txt')
    val_txt = os.path.join(args.dst, 'scannetv2_val.txt')
    test_txt = os.path.join(args.dst, 'scannetv2_test.txt')

    if not os.path.exists(train_txt):
        shutil.copy(os.path.join(args.src, 'scannetv2_train.txt'), train_txt)
    if not os.path.exists(val_txt):
        shutil.copy(os.path.join(args.src, 'scannetv2_val.txt'), val_txt)
    if not os.path.exists(test_txt):
        shutil.copy(os.path.join(args.src, 'scannetv2_test.txt'), test_txt)

    with open(train_txt, 'r') as fp:
        train_scenes = [f.strip() for f in fp.readlines()]
    with open(val_txt, 'r') as fp:
        val_scenes = [f.strip() for f in fp.readlines()]
    with open(test_txt, 'r') as fp:
        test_scenes = ([f.strip() for f in fp.readlines()])

    test_dir_src = os.path.join(args.src, 'scans_test')
    test_dir_dst = os.path.join(args.dst, 'scans_test')
    trainval_dir_src = os.path.join(args.src, 'scans')
    trainval_dir_dst = os.path.join(args.dst, 'scans')

    for i, scene_name in enumerate(train_scenes):
        print('{} / {}: {}'.format(i+1,
                                   len(train_scenes)+len(val_scenes)+len(test_scenes), scene_name))
        scene_dir_src = os.path.join(trainval_dir_src, scene_name)
        scene_dir_dst = os.path.join(trainval_dir_dst, scene_name)
        process_scene(scene_dir_src, scene_dir_dst)

    for i, scene_name in enumerate(val_scenes):
        print('{} / {}: {}'.format(i+1+len(train_scenes),
                                   len(train_scenes)+len(val_scenes)+len(test_scenes), scene_name))
        scene_dir_src = os.path.join(trainval_dir_src, scene_name)
        scene_dir_dst = os.path.join(trainval_dir_dst, scene_name)
        process_scene(scene_dir_src, scene_dir_dst)

    for i, scene_name in enumerate(test_scenes):
        print('{} / {}: {}'.format(i+1+len(train_scenes)+len(val_scenes),
                                   len(train_scenes)+len(val_scenes)+len(test_scenes), scene_name))
        scene_dir_src = os.path.join(test_dir_src, scene_name)
        scene_dir_dst = os.path.join(test_dir_dst, scene_name)
        process_scene(scene_dir_src, scene_dir_dst)
