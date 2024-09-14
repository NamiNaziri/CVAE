import os
import sys

import joblib
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

from mlexp_utils import my_logging
from mlexp_utils.dirs import proj_dir
from phc.utils.motion_lib_base import compute_motion_dof_vels
from poselib.poselib.skeleton.skeleton3d import (
    SkeletonState,
    SkeletonTree,
    SkeletonMotion,
)
from viz.visual_data_pv import XMLVisualDataContainer

sys.path.append(os.path.abspath("./src"))

from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pyvista as pv
import rerun as rr


def main():
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

    rr.init(args.out_name, recording_id=args.out_name)

    current_path = os.getcwd()
    print("Current Path:", current_path)

    pkl_file_path = f"{proj_dir}/data/nami/our/martial_arts.pkl"
    with open(pkl_file_path, "rb") as f:
        data = joblib.load(f)

    # Reconstruction visual dumping
    sk_tree = SkeletonTree.from_mjcf("phc/data/assets/mjcf/smpl_humanoid_1.xml")
    gt_visual_data = XMLVisualDataContainer("phc/data/assets/mjcf/my_smpl_humanoid.xml")
    recon_visual_data = XMLVisualDataContainer(
        "phc/data/assets/mjcf/my_smpl_humanoid.xml"
    )
    pl = pv.Plotter(off_screen=True, window_size=(608, 608))
    pl.add_mesh(gt_visual_data.plane)
    pl.add_axes()
    gt_actors = []
    for mesh, ax in zip(gt_visual_data.meshes, gt_visual_data.axes):
        actor = pl.add_mesh(mesh, color="blue")
        gt_actors.append(actor)

    recon_actors = []
    for mesh, ax in zip(recon_visual_data.meshes, recon_visual_data.axes):
        actor = pl.add_mesh(mesh, color="red")
        recon_actors.append(actor)

    pl.enable_shadows()

    for k, v in tqdm(d.items()):
        # trans = torch.as_tensor(d[k]["trans_orig"])
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
        rb_rot_sixd = (
            Rotation.from_quat(rb_rot.reshape(-1, 4))
            .as_matrix()
            .reshape((*rb_rot.shape[:-1], 3, 3))[..., :2]
            .reshape((*rb_rot.shape[:-1], 6))
        )
        rb_vel = curr_motion.global_velocity.cpu().numpy()
        rb_ang = curr_motion.global_angular_velocity.cpu().numpy()
        dof_vel = curr_motion.curr_dof_vels.cpu().numpy()

        rb_poses.append(rb_pos.reshape(rb_pos.shape[0], -1))
        rb_rot_sixds.append(rb_rot_sixd.reshape(rb_rot_sixd.shape[0], -1))
        rb_vels.append(rb_vel.reshape(rb_vel.shape[0], -1))
        rb_angs.append(rb_ang.reshape(rb_ang.shape[0], -1))
        dof_vels.append(dof_vel.reshape(dof_vel.shape[0], -1))

    rb_poses = np.concatenate(rb_poses, axis=0)
    rb_rot_sixds = np.concatenate(rb_rot_sixds, axis=0)
    rb_vels = np.concatenate(rb_vels, axis=0)
    rb_angs = np.concatenate(rb_angs, axis=0)
    dof_vels = np.concatenate(dof_vels, axis=0)

    # out_path = f"{out_dir}/torchready_v3.pkl"
    out_path = f"{out_dir}/torchready_v2.pkl"
    with open(out_path, "wb") as f:
        torch.save(
            {
                "rb_pos": rb_poses,
                "rb_rot_sixd": rb_rot_sixds,
                "rb_vel": rb_vels,
                "rb_ang": rb_angs,
                "dof_vel": dof_vels,
            },
            f,
            pickle_protocol=4,
        )


    img_rows = []
    for delta_v in np.linspace(-0.5, 0.5, 5)[::-1]:
        imgs = []
        for t in tqdm(range(0, 10)):
            rb_pos_reshaped = rb_pos[selected_valid_idxs[t]].reshape(24, 3)
            rb_rot_sixd_reshaped = rb_rot_sixd[selected_valid_idxs[t]].reshape(24, 3, 2)
            third_column = np.cross(
                rb_rot_sixd_reshaped[..., 0], rb_rot_sixd_reshaped[..., 1], axis=-1
            )
            rb_rot_rotmat = np.concatenate(
                [rb_rot_sixd_reshaped, third_column[..., None]], axis=-1
            )
            rb_rot_quat = Rotation.from_matrix(rb_rot_rotmat).as_quat()
            sk_state = SkeletonState.from_rotation_and_root_translation(
                sk_tree,
                torch.as_tensor(rb_rot_quat, dtype=torch.float),
                torch.as_tensor(rb_pos_reshaped[0], dtype=torch.float),
                is_local=False,
            )
            # gt_curr_motion = SkeletonMotion.from_skeleton_state(sk_state, 30)
            # gt_curr_dof_vels = compute_motion_dof_vels(gt_curr_motion)
            # gt_curr_motion.curr_dof_vels = gt_curr_dof_vels
            orig_x = xs[[t]]
            orig_y = ys[[t]]
            mutated_y = orig_y * 1
            mutated_y[..., 9] += float(delta_v)
            _, decoded, mu, log_var = cvae.forward(
                orig_x,
                mutated_y,
                train_yes=False,
            )
            # rb_rot_sixd_recon = decoded.cpu().detach().numpy()
            # muys = torch.cat([mu, mutated_y], dim=-1)
            # rb_rot_sixd_recon = cvae.decoder(muys)
            z = cvae.reparameterize(mu, log_var)
            # rb_rot_sixd_recon = cvae.decode(z, cvae.cond_rms.normalize(mutated_y))
            rb_rot_sixd_recon = cvae.decode(z, mutated_y)
            rb_rot_sixd_recon = rb_rot_sixd_recon.cpu().detach().numpy()

            recon_rot_sixd_reshaped = rb_rot_sixd_reshaped * 1
            recon_rot_sixd_reshaped[1:] = rb_rot_sixd_recon[0].reshape(23, 3, 2)
            third_column = np.cross(
                recon_rot_sixd_reshaped[..., 0],
                recon_rot_sixd_reshaped[..., 1],
                axis=-1,
            )
            recon_rot_rotmat = np.concatenate(
                [recon_rot_sixd_reshaped, third_column[..., None]], axis=-1
            )
            recon_rot_quat = Rotation.from_matrix(recon_rot_rotmat).as_quat()
            mutated_trans = rb_pos_reshaped[0] * 1
            mutated_trans[2] += float(delta_v)
            recon_sk_state = SkeletonState.from_rotation_and_root_translation(
                sk_tree,
                torch.as_tensor(recon_rot_quat, dtype=torch.float),
                torch.as_tensor(mutated_trans, dtype=torch.float),
                is_local=False,
            )
            # recon_curr_motion = SkeletonMotion.from_skeleton_state(sk_state, 30)
            # recon_curr_dof_vels = compute_motion_dof_vels(recon_curr_motion)
            # recon_curr_motion.curr_dof_vels = recon_curr_dof_vels
            gt_global_translation = sk_state.global_translation.detach().cpu().numpy()
            gt_global_rotation = sk_state.global_rotation.detach().cpu().numpy()

            recon_global_translation = (
                recon_sk_state.global_translation.detach().cpu().numpy()
            )
            recon_global_rotation = (
                recon_sk_state.global_rotation.detach().cpu().numpy()
            )

            for i in range(len(gt_actors)):
                gt_actor = gt_actors[i]
                recon_actor = recon_actors[i]
                m = np.eye(4)
                pos = gt_global_translation[i] * 1
                gt_rotmat = Rotation.from_quat(gt_global_rotation[i]).as_matrix()
                # pos = rb_pos_reshaped[i] * 1
                # gt_rotmat = rb_rot_rotmat[i]
                m[:3, :3] = gt_rotmat
                m[:3, 3] = pos
                gt_actor.user_matrix = m

                m = np.eye(4)
                recon_rotmat = Rotation.from_quat(recon_global_rotation[i]).as_matrix()
                pos = recon_global_translation[i] * 1
                m[:3, :3] = recon_rotmat
                m[:3, 3] = pos
                recon_actor.user_matrix = m

            distance = 5
            pl.camera.position = (-distance, -distance, 4)
            pl.camera.focal_point = (0, 0, 0)
            # pl.camera.up = (0, 0, 1)
            pl.render()
            # pl.show()

            img = pl.screenshot()
            # plt.figure()
            # plt.imshow(img)
            # plt.show()
            imgs.append(img)

        img_row = np.concatenate(imgs, axis=1)
        rr.set_time_seconds("DeltaZ", delta_v)
        rr.log(f"ImageRow/GSE{global_steps_elapsed:06d}", rr.Image(img_row))
        # rr.log(f"DeltaZ={delta_v}", rr.Image(img_row))
        img_rows.append(img_row)
    img = np.concatenate(img_rows, axis=0)
    if args.debug_yes:
        plt.figure()
        plt.imshow(img)
        plt.show()
        plt.close()

    # rr.log("ReconEvalImage", rr.Image(img))

    rr.save(f"{logdir}/{global_steps_elapsed:08d}.rrd")

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--debug_yes", action="store_true")
    args = parser.parse_args()

    main()
