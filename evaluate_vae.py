import os
import sys

import imageio
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
import ast
import inspect
import numpy as np
import torch.distributions as td
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
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

    writer = SummaryWriter(logdir)

    rr.init(args.out_name, recording_id=args.out_name)

    vae_dict = torch.load(args.model_path)
    for line in vae_dict["imports"].split("\n"):
        exec(line)
    exec(vae_dict["model_src"])
    vae = eval(vae_dict["model_cls_name"])(
        *vae_dict["model_args"], **vae_dict["model_kwargs"]
    )
    vae.load_state_dict(vae_dict["model_state_dict"])
    vae.requires_grad_(False)
    vae = vae.to("cuda")

    global_steps_elapsed = vae_dict["global_steps_elapsed"]
    rr.set_time_sequence("GlobalStepsElapsed", global_steps_elapsed)

    current_path = os.getcwd()
    print("Current Path:", current_path)

    pkl_file_path = f"{proj_dir}/data/nami/torchready/torchready_v2.pkl"
    data = torch.load(pkl_file_path)
    rb_rot_sixd = data["rb_rot_sixd"]
    rb_pos = data["rb_pos"]

    n_train = int(rb_pos.shape[0] * 0.90)
    n_valid = rb_pos.shape[0] - n_train

    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    train_idxs = np.random.choice(rb_pos.shape[0], n_train, replace=False)
    valid_idxs = np.setdiff1d(np.arange(rb_pos.shape[0]), train_idxs)

    selected_train_idxs = np.random.choice(train_idxs, 10, replace=False)
    selected_valid_idxs = np.random.choice(valid_idxs, 10, replace=False)

    # _, rb_rot_sixd_recon, _, _ = vae.forward(
    #     torch.as_tensor(rb_rot_sixd[:10], dtype=torch.float, device="cuda"),
    #     train_yes=False,
    # )
    _, _, mu, _ = vae.forward(
        torch.as_tensor(
            rb_rot_sixd[selected_valid_idxs], dtype=torch.float, device="cuda"
        ),
        train_yes=False,
    )
    rb_rot_sixd_recon = vae.decoder(mu)
    rb_rot_sixd_recon = rb_rot_sixd_recon.cpu().detach().numpy()

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

        recon_rot_sixd_reshaped = rb_rot_sixd_recon[t].reshape(24, 3, 2)
        third_column = np.cross(
            recon_rot_sixd_reshaped[..., 0], recon_rot_sixd_reshaped[..., 1], axis=-1
        )
        recon_rot_rotmat = np.concatenate(
            [recon_rot_sixd_reshaped, third_column[..., None]], axis=-1
        )
        recon_rot_quat = Rotation.from_matrix(recon_rot_rotmat).as_quat()
        recon_sk_state = SkeletonState.from_rotation_and_root_translation(
            sk_tree,
            torch.as_tensor(recon_rot_quat, dtype=torch.float),
            torch.as_tensor(rb_pos_reshaped[0], dtype=torch.float),
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
        recon_global_rotation = recon_sk_state.global_rotation.detach().cpu().numpy()

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

    img_row_1 = np.concatenate(imgs[:5], axis=1)
    img_row_2 = np.concatenate(imgs[5:], axis=1)
    img = np.concatenate([img_row_1, img_row_2], axis=0)

    rr.log("ReconEvalImage", rr.Image(img))

    # VAE sampling image dumping
    noise = torch.randn(10, vae.hammer_size, dtype=torch.float, device="cuda")
    rb_rot_sixd_samp = vae.decoder(noise)
    rb_rot_sixd_samp = rb_rot_sixd_samp.cpu().detach().numpy()

    # Reconstruction visual dumping
    sk_tree = SkeletonTree.from_mjcf("phc/data/assets/mjcf/smpl_humanoid_1.xml")
    samp_visual_data = XMLVisualDataContainer(
        "phc/data/assets/mjcf/my_smpl_humanoid.xml"
    )
    pl = pv.Plotter(off_screen=True, window_size=(608, 608))
    pl.add_mesh(gt_visual_data.plane)
    pl.add_axes()
    samp_actors = []
    for mesh, ax in zip(samp_visual_data.meshes, samp_visual_data.axes):
        actor = pl.add_mesh(mesh, color="green")
        samp_actors.append(actor)

    pl.enable_shadows()

    imgs = []
    for t in tqdm(range(0, 10)):
        samp_rot_sixd_reshaped = rb_rot_sixd_samp[t].reshape(24, 3, 2)
        third_column = np.cross(
            samp_rot_sixd_reshaped[..., 0], samp_rot_sixd_reshaped[..., 1], axis=-1
        )
        samp_rot_rotmat = np.concatenate(
            [samp_rot_sixd_reshaped, third_column[..., None]], axis=-1
        )
        samp_rot_quat = Rotation.from_matrix(samp_rot_rotmat).as_quat()
        samp_sk_state = SkeletonState.from_rotation_and_root_translation(
            sk_tree,
            torch.as_tensor(samp_rot_quat, dtype=torch.float),
            torch.as_tensor(rb_pos_reshaped[0], dtype=torch.float),
            is_local=False,
        )
        samp_global_translation = (
            samp_sk_state.global_translation.detach().cpu().numpy()
        )
        samp_global_rotation = samp_sk_state.global_rotation.detach().cpu().numpy()

        for i in range(len(samp_actors)):
            samp_actor = samp_actors[i]
            m = np.eye(4)
            samp_rotmat = Rotation.from_quat(samp_global_rotation[i]).as_matrix()
            pos = samp_global_translation[i] * 1
            m[:3, :3] = samp_rotmat
            m[:3, 3] = pos
            samp_actor.user_matrix = m

        distance = 5
        pl.camera.position = (-distance, -distance, 4)
        pl.camera.focal_point = (0, 0, 0)
        # pl.camera.up = (0, 0, 1)
        pl.render()
        # pl.show()

        img = pl.screenshot()
        imgs.append(img)

    img_row_1 = np.concatenate(imgs[:5], axis=1)
    img_row_2 = np.concatenate(imgs[5:], axis=1)
    img = np.concatenate([img_row_1, img_row_2], axis=0)
    # plt.figure()
    # plt.imshow(img)
    # plt.show()
    # plt.close()

    rr.log("SampleEvalImage", rr.Image(img))

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
