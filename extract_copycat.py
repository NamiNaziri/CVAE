from phc.utils.motion_lib_base import compute_motion_dof_vels
from poselib.poselib.skeleton.skeleton3d import (
    SkeletonTree,
    SkeletonState,
    SkeletonMotion,
)
import os
import sys

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from mlexp_utils import my_logging
from mlexp_utils.dirs import proj_dir

sys.path.append(os.path.abspath("./src"))

from argparse import ArgumentParser
import debugpy

debugpy.listen(5679)
from phc.utils import torch_utils
def _local_rotation_to_dof_smpl( local_rot):
        B, J, _ = local_rot.shape
        dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)

def main(args):
    if args.debug_yes:
        import pydevd_pycharm

        pydevd_pycharm.settrace(
            "localhost",
            port=12345,
            stdoutToServer=True,
            stderrToServer=True,
            suspend=False,
        )

    run_dir = os.getcwd()
    out_dir = f"{proj_dir}/out/{args.run_name}/{args.out_name}"
    os.makedirs(out_dir, exist_ok=True)

    logdir = f"{proj_dir}/logdir/{args.run_name}/{args.out_name}"
    os.makedirs(logdir, exist_ok=True)
    logger = my_logging.get_logger(f"{args.out_name}", logdir)
    logger.info(f"Starting")

    data_dir = f"None/data/nami/amass/pkls"
    # copycat_path = f"{data_dir}/amass_copycat_take5_train.pkl"
    # copycat_path = f"./amass_copycat_take5_train_small.pkl"
    copycat_path = os.path.join(data_dir, f"amass_copycat_take5_train.pkl")
    with open(copycat_path, "rb") as f:
        d = joblib.load(f)
    # d = torch.load(copycat_path)
    logger.info(f"Loaded copycat data from {copycat_path}")

    # dd = {}
    # for i in range(100):
    #     dd[list(d.keys())[i]] = d[list(d.keys())[i]]
    #     with open(f"./amass_copycat_take5_train_medium.pkl", "wb") as f:
    #         joblib.dump(dd, f)

    rb_poses = []
    rb_rot_sixds = []
    rb_rots=[]
    root_rots=[]
    dof_vels = []
    dof_poses=[]
    rb_vels = []
    rb_angs = []
    rb_pos_invs = []
    rb_vel_invs = []
    rb_rot_sixd_invs = []
    sk_tree = SkeletonTree.from_mjcf(f"{proj_dir}/data/nami/mjcf/smpl_humanoid_1.xml")
    circle_list = [
        "0-BioMotionLab_NTroje_rub042_0027_circle_walk_poses",
    ]

    for k in d.keys():
        if "circle" in k or "Circle" in k:
            circle_list.append(k)

    newD = {}
    for c in circle_list:
        newD[c] = d[c]
    d = newD

    # d = {
    #     "0-BioMotionLab_NTroje_rub042_0027_circle_walk_poses": d[
    #         "0-BioMotionLab_NTroje_rub042_0027_circle_walk_poses"
    #     ]
    # }
    # sum = 0
    # sum_map = {}
    # sum = 0
    # for index, (k, v) in tqdm(enumerate(d.items()), total=len(d)):
    #     start = sum * 1
    #     sum += d[k]["root_trans_offset"].shape[0]
    #     sum_map[sum] = k
    #     print(f"{k}: start: {start}  end: {sum}")
    # torch.save(sum_map, "sum_map.pt")
    # sum = 0
    # with open("output2.txt", "w") as f:
    #     for index, (k, v) in tqdm(enumerate(d.items()), total=len(d)):
    #         start = sum * 1
    #         sum += d[k]["root_trans_offset"].shape[0]
    #         f.write(f"{k}: start: {start}  end: {sum}  duration: {sum-start}\n")

    # for index, (k, v) in tqdm(enumerate(d.items()), total=len(d)):
    #     start = sum * 1
    #     sum += d[k]["root_trans_offset"].shape[0]
    #     print(f"{k}: start: {start}  end: {sum}")
    # a
    for index, (k, v) in tqdm(enumerate(d.items()), total=len(d)):
        # if index < 4:
        #     continue
        # if index > 1000:
        #     break
        trans = torch.as_tensor(d[k]["root_trans_offset"])
        pose_quat_global = torch.tensor(d[k]["pose_quat_global"])
        sk_state = SkeletonState.from_rotation_and_root_translation(
            sk_tree, pose_quat_global, trans, is_local=False
        )
        curr_motion = SkeletonMotion.from_skeleton_state(sk_state, 30)
        curr_dof_vels = compute_motion_dof_vels(curr_motion)
        curr_motion.curr_dof_vels = curr_dof_vels

        rb_pos = curr_motion.global_translation.cpu().numpy()
        rb_rot = curr_motion.global_rotation.cpu().numpy()
        root_rot = rb_rot[..., 0, :] * 1
        correction = Rotation.from_euler(
            "Z",
            -Rotation.from_quat(rb_rot[:, 0])
            .as_euler("zyx")[..., [0]]
            .repeat(rb_rot.shape[1], -1)
            .reshape(-1),
        )  # undo root rotation from all body parts of all frames

        rb_pos_inv = rb_pos * 1
        rb_pos_inv[..., :2] = rb_pos[..., :2] - rb_pos[:, [0], :2]  # center to (0, 0)
        rb_pos_inv = correction.apply(rb_pos_inv.reshape(-1, 3)).reshape((-1, 24, 3))

        rb_scipy_rot = Rotation.from_quat(rb_rot.reshape(-1, 4))
        rb_scipy_rot_inv = correction * rb_scipy_rot

        rb_rot_sixd = (
            rb_scipy_rot.as_matrix()
            .reshape((*rb_rot.shape[:-1], 3, 3))[..., :2]
            .reshape((*rb_rot.shape[:-1], 6))
        )
        rb_rot_sixd_inv = (
            rb_scipy_rot_inv.as_matrix()
            .reshape((*rb_rot.shape[:-1], 3, 3))[..., :2]
            .reshape((*rb_rot.shape[:-1], 6))
        )
        rb_vel = curr_motion.global_velocity.cpu().numpy()

        rb_vel_inv = rb_vel * 1
        rb_vel_inv = correction.apply(rb_vel_inv.reshape(-1, 3)).reshape((-1, 24, 3))

        rb_ang = curr_motion.global_angular_velocity.cpu().numpy()
        dof_vel = curr_motion.curr_dof_vels.cpu().numpy()

        dof_pos = _local_rotation_to_dof_smpl(curr_motion.local_rotation).cpu().numpy()

        rb_poses.append(rb_pos.reshape(rb_pos.shape[0], -1))
        rb_rots.append(rb_rot.reshape(rb_rot.shape[0],-1))
        root_rots.append(root_rot.reshape(root_rot.shape[0],-1))
        rb_pos_invs.append(rb_pos_inv.reshape(rb_pos_inv.shape[0], -1))
        rb_rot_sixds.append(rb_rot_sixd.reshape(rb_rot_sixd.shape[0], -1))
        rb_rot_sixd_invs.append(rb_rot_sixd_inv.reshape(rb_rot_sixd_inv.shape[0], -1))
        rb_vels.append(rb_vel.reshape(rb_vel.shape[0], -1))
        rb_vel_invs.append(rb_vel_inv.reshape(rb_vel_inv.shape[0], -1))
        rb_angs.append(rb_ang.reshape(rb_ang.shape[0], -1))
        dof_vels.append(dof_vel.reshape(dof_vel.shape[0], -1))
        dof_poses.append(dof_pos.reshape(dof_pos.shape[0], -1))

    rb_poses = np.concatenate(rb_poses, axis=0)
    rb_rots = np.concatenate(rb_rots, axis=0)
    root_rots =  np.concatenate(root_rots, axis=0)
    rb_pos_invs = np.concatenate(rb_pos_invs, axis=0)
    rb_rot_sixds = np.concatenate(rb_rot_sixds, axis=0)
    rb_rot_sixd_invs = np.concatenate(rb_rot_sixd_invs, axis=0)
    rb_vels = np.concatenate(rb_vels, axis=0)
    rb_vel_invs = np.concatenate(rb_vel_invs, axis=0)
    rb_angs = np.concatenate(rb_angs, axis=0)
    dof_vels = np.concatenate(dof_vels, axis=0)
    dof_poses = np.concatenate(dof_poses, axis=0)

    # invariant rb_rot_sixd
    # rb_rot_sixd_reshaped = rb_rot_sixds.reshape(-1, 24, 3, 2)
    # third_column = np.cross(
    #     rb_rot_sixd_reshaped[..., 0], rb_rot_sixd_reshaped[..., 1], axis=-1
    # )
    # rb_rot_rotmat = np.concatenate(
    #     [rb_rot_sixd_reshaped, third_column[..., None]], axis=-1
    # )
    # rb_rot_euler = Rotation.from_matrix(rb_rot_rotmat.reshape(-1, 3, 3)).as_euler("zyx")
    # rb_rot_euler.reshape(*rb_rot_sixd_reshaped.shape[:-2], 3)[
    #     :, :, 0
    # ] -= rb_rot_euler.reshape(*rb_rot_sixd_reshaped.shape[:-2], 3)[:, [0], 0]
    # # TODO: this is wrong! change it to something smilar as train_cvae_light or enjoy_cvae_light
    # rb_rot_sixd_inv = (
    #     Rotation.from_euler("zyx", rb_rot_euler)
    #     .as_matrix()
    #     .reshape((*rb_rot_sixd_reshaped.shape[:-2], 3, 3))[..., :2]
    #     .reshape((*rb_rot_sixd_reshaped.shape[:-2], 6))
    # )

    # out_path = f"{out_dir}/torchready_v3.pkl"
    out_path = f"torchready_v5_dof_1k.pkl"
    with open(out_path, "wb") as f:
        torch.save(
            {
                "rb_pos": rb_poses,
                "rb_pos_inv": rb_pos_invs,
                "rb_rot": rb_rots,
                "root_rot": root_rots,
                "rb_rot_sixd": rb_rot_sixds,
                "rb_rot_sixd_inv": rb_rot_sixd_invs,
                "rb_vel": rb_vels,
                "rb_vel_inv": rb_vel_invs,
                "rb_ang": rb_angs,
                "dof_vel": dof_vels,
                "dof_pos": dof_poses
            },
            f,
            pickle_protocol=4,
        )

    # visual_data = XMLVisualDataContainer("phc/data/assets/mjcf/my_smpl_humanoid.xml")
    # pl = pv.Plotter(off_screen=True, window_size=(608, 608))
    # # pl = pv.Plotter(off_screen=False, window_size=(608, 608))
    # actors = []
    # pl.add_mesh(visual_data.plane)
    # pl.add_axes()
    # for mesh, ax in zip(visual_data.meshes, visual_data.axes):
    #     actor = pl.add_mesh(mesh, color="red")
    #     actors.append(actor)
    #
    # target_actors = []
    # for i in range(24):
    #     actor = pl.add_mesh(visual_data.targets[i], color="red")
    #     target_actors.append(actor)
    #
    # pl.enable_shadows()
    #
    # imgs = []
    # for t in tqdm(range(0, rb_pos.shape[1], 2)):
    #     for i, actor in enumerate(actors):
    #         m = np.eye(4)
    #         pos = rb_pos[0, t, i] * 1
    #         rb_rot_sixd_reshaped = rb_rot_sixd[0, t, i].reshape(3, 2)
    #         third_column = np.cross(rb_rot_sixd_reshaped[:, 0], rb_rot_sixd_reshaped[:, 1])
    #         rb_rot_rotmat = np.concatenate([rb_rot_sixd_reshaped, third_column[:, None]], axis=1)
    #         m[:3, :3] = rb_rot_rotmat
    #         m[:3, 3] = pos
    #         actor.user_matrix = m
    #
    #     distance = 5
    #     pl.camera.position = (-distance, -distance, 4)
    #     pl.camera.focal_point = (0, 0, 0)
    #     # pl.camera.up = (0, 0, 1)
    #     pl.render()
    #     # pl.show()
    #
    #     # plt.figure()
    #     # img = pl.screenshot()
    #     # plt.imshow(img)
    #     # plt.show()
    #     img = np.array(pl.screenshot())
    #     imgs.append(img)
    #
    # w = imageio.get_writer(
    #     f"{proj_dir}/out/{args.run_name}/{args.out_name}/movie1.mp4",
    #     format="FFMPEG",
    #     mode="I",
    #     fps=15,
    #     codec="h264",
    #     pixelformat="yuv420p",
    # )
    # for img in imgs:
    #     w.append_data(img)
    # w.close()

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--debug_yes", action="store_true")
    args = parser.parse_args()

    main(args)
