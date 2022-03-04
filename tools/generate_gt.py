# from NeuralRecon:
# https://github.com/zju3dv/NeuralRecon/blob/master/tools/tsdf_fusion/generate_gt.py

#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
# 
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
# 
#    1. Definitions.
# 
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
# 
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
# 
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
# 
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
# 
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
# 
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
# 
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
# 
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
# 
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
# 
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
# 
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
# 
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
# 
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
# 
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
# 
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
# 
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
# 
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
# 
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
# 
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
# 
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
# 
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
# 
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
# 
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
# 
#    END OF TERMS AND CONDITIONS
# 
#    APPENDIX: How to apply the Apache License to your work.
# 
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
# 
#    Copyright [yyyy] [name of copyright owner]
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import sys
sys.path.append('.')

import time
import pickle
import argparse
from tqdm import tqdm
import ray
import torch.multiprocessing
from tools.simple_loader import *

from vortx.tsdf_fusion import *

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Fuse ground truth tsdf')
    parser.add_argument("--dataset", default='scannet')
    parser.add_argument("--data_path", metavar="DIR",
                        help="path to raw dataset", default='/data/scannet/output/')
    parser.add_argument("--save_name", metavar="DIR",
                        help="file name", default='all_tsdf')
    parser.add_argument('--test', action='store_true',
                        help='prepare the test set')
    parser.add_argument('--max_depth', default=3., type=float,
                        help='mask out large depth values since they are noisy')
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--margin', default=3, type=int)
    parser.add_argument('--voxel_size', default=0.04, type=float)

    parser.add_argument('--window_size', default=9, type=int)
    parser.add_argument('--min_angle', default=15, type=float)
    parser.add_argument('--min_distance', default=0.1, type=float)

    # ray multi processes
    parser.add_argument('--n_proc', type=int, default=16, help='#processes launched to process scenes.')
    parser.add_argument('--n_gpu', type=int, default=2, help='#number of gpus')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--loader_num_workers', type=int, default=8)
    return parser.parse_args()


args = parse_args()
args.save_path = os.path.join(args.data_path, args.save_name)


def save_tsdf_full(args, scene_path, cam_intr, depth_list, cam_pose_list, color_list, save_mesh=False):
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    vol_bnds = np.zeros((3, 2))

    n_imgs = len(depth_list.keys())
    if n_imgs > 200:
        ind = np.linspace(0, n_imgs - 1, 200).astype(np.int32)
        image_id = np.array(list(depth_list.keys()))[ind]
    else:
        image_id = depth_list.keys()
    for id in image_id:
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]

        # Compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    # ======================================================================================================== #

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol_list = []
    for l in range(args.num_layers):
        tsdf_vol_list.append(TSDFVolume(vol_bnds, voxel_size=args.voxel_size * 2 ** l, margin=args.margin))

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for id in depth_list.keys():
        if id % 100 == 0:
            print("{}: Fusing frame {}/{}".format(scene_path, str(id), str(n_imgs)))
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]
        if len(color_list) == 0:
            color_image = None
        else:
            color_image = color_list[id]

        # Integrate observation into voxel volume (assume color aligned with depth)
        for l in range(args.num_layers):
            tsdf_vol_list[l].integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    tsdf_info = {
        'vol_origin': tsdf_vol_list[0]._vol_origin,
        'voxel_size': tsdf_vol_list[0]._voxel_size,
    }
    tsdf_path = os.path.join(args.save_path, scene_path)
    if not os.path.exists(tsdf_path):
        os.makedirs(tsdf_path)

    with open(os.path.join(args.save_path, scene_path, 'tsdf_info.pkl'), 'wb') as f:
        pickle.dump(tsdf_info, f)

    for l in range(args.num_layers):
        tsdf_vol, color_vol, weight_vol = tsdf_vol_list[l].get_volume()
        np.savez_compressed(os.path.join(args.save_path, scene_path, 'full_tsdf_layer{}'.format(str(l))), tsdf_vol)

    if save_mesh:
        for l in range(args.num_layers):
            print("Saving mesh to mesh{}.ply...".format(str(l)))
            verts, faces, norms, colors = tsdf_vol_list[l].get_mesh()

            meshwrite(os.path.join(args.save_path, scene_path, 'mesh_layer{}.ply'.format(str(l))), verts, faces, norms,
                      colors)

            # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
            # print("Saving point cloud to pc.ply...")
            # point_cloud = tsdf_vol_list[l].get_point_cloud()
            # pcwrite(os.path.join(args.save_path, scene_path, 'pc_layer{}.ply'.format(str(l))), point_cloud)


def save_fragment_pkl(args, scene, cam_intr, depth_list, cam_pose_list):
    fragments = []
    print('segment: process scene {}'.format(scene))

    # gather pose
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = np.inf
    vol_bnds[:, 1] = -np.inf

    all_ids = []
    ids = []
    all_bnds = []
    count = 0
    last_pose = None
    for id in depth_list.keys():
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]

        if count == 0:
            ids.append(id)
            vol_bnds = np.zeros((3, 2))
            vol_bnds[:, 0] = np.inf
            vol_bnds[:, 1] = -np.inf
            last_pose = cam_pose
            # Compute camera view frustum and extend convex hull
            view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
            count += 1
        else:
            angle = np.arccos(
                ((np.linalg.inv(cam_pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                    [0, 0, 1])).sum())
            dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
            if angle > (args.min_angle / 180) * np.pi or dis > args.min_distance:
                ids.append(id)
                last_pose = cam_pose
                # Compute camera view frustum and extend convex hull
                view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
                vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
                vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
                count += 1
                if count == args.window_size:
                    all_ids.append(ids)
                    all_bnds.append(vol_bnds)
                    ids = []
                    count = 0

    with open(os.path.join(args.save_path, scene, 'tsdf_info.pkl'), 'rb') as f:
        tsdf_info = pickle.load(f)

    # save fragments
    for i, bnds in enumerate(all_bnds):
        if not os.path.exists(os.path.join(args.save_path, scene, 'fragments', str(i))):
            os.makedirs(os.path.join(args.save_path, scene, 'fragments', str(i)))
        fragments.append({
            'scene': scene,
            'fragment_id': i,
            'image_ids': all_ids[i],
            'vol_origin': tsdf_info['vol_origin'],
            'voxel_size': tsdf_info['voxel_size'],
        })

    with open(os.path.join(args.save_path, scene, 'fragments.pkl'), 'wb') as f:
        pickle.dump(fragments, f)


@ray.remote(num_cpus=args.num_workers + 1, num_gpus=(1 / args.n_proc))
def process_with_single_worker(args, scannet_files):
    for scene in tqdm(scannet_files):
        if os.path.exists(os.path.join(args.save_path, scene, 'fragments.pkl')):
            continue
        print('read from disk')

        depth_all = {}
        cam_pose_all = {}
        color_all = {}

        if args.dataset == 'scannet':
            n_imgs = len(os.listdir(os.path.join(args.data_path, scene, 'color')))
            intrinsic_dir = os.path.join(args.data_path, scene, 'intrinsic', 'intrinsic_depth.txt')
            cam_intr = np.loadtxt(intrinsic_dir, delimiter=' ')[:3, :3]
            dataset = ScanNetDataset(n_imgs, scene, args.data_path, args.max_depth)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, collate_fn=collate_fn,
                                                 batch_sampler=None, num_workers=args.loader_num_workers)

        for id, (cam_pose, depth_im, _) in enumerate(dataloader):
            if id % 100 == 0:
                print("{}: read frame {}/{}".format(scene, str(id), str(n_imgs)))

            if cam_pose[0][0] == np.inf or cam_pose[0][0] == -np.inf or cam_pose[0][0] == np.nan:
                continue
            depth_all.update({id: depth_im})
            cam_pose_all.update({id: cam_pose})
            # color_all.update({id: color_image})

        save_tsdf_full(args, scene, cam_intr, depth_all, cam_pose_all, color_all, save_mesh=False)
        save_fragment_pkl(args, scene, cam_intr, depth_all, cam_pose_all)


def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret


def generate_pkl(args):
    all_scenes = sorted(os.listdir(args.save_path))
    # todo: fix for both train/val/test
    if not args.test:
        splits = ['train', 'val']
    else:
        splits = ['test']
    for split in splits:
        fragments = []
        with open(os.path.join(args.save_path, 'splits', 'scannetv2_{}.txt'.format(split))) as f:
            split_files = f.readlines()
        for scene in all_scenes:
            if 'scene' not in scene:
                continue
            if scene + '\n' in split_files:
                with open(os.path.join(args.save_path, scene, 'fragments.pkl'), 'rb') as f:
                    frag_scene = pickle.load(f)
                fragments.extend(frag_scene)

        with open(os.path.join(args.save_path, 'fragments_{}.pkl'.format(split)), 'wb') as f:
            pickle.dump(fragments, f)


if __name__ == "__main__":
    all_proc = args.n_proc * args.n_gpu

    ray.init(num_cpus=all_proc * (args.num_workers + 1), num_gpus=args.n_gpu)

    if args.dataset == 'scannet':
        if not args.test:
            args.data_path = os.path.join(args.data_path, 'scans')
        else:
            args.data_path = os.path.join(args.data_path, 'scans_test')
        files = sorted(os.listdir(args.data_path))
    else:
        raise NameError('error!')

    files = split_list(files, all_proc)

    ray_worker_ids = []
    for w_idx in range(all_proc):
        ray_worker_ids.append(process_with_single_worker.remote(args, files[w_idx]))

    results = ray.get(ray_worker_ids)

    # if args.dataset == 'scannet':
    #     generate_pkl(args)
