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
from dearpygui import dearpygui as dpg


def visualize_dpg_controller(
    logger,
    cvae,
):

    def exit_callback(sender, app_data, user_data):
        dpg.destroy_context()
        return

    def slider_callback(sender, app_data, user_data):
        logger.info(f"Slider callback start")
        logger.info(f"Slider callback done")

    dpg.create_context()

    if args.debug_yes:
        dpg.configure_app(manual_callback_management=True)

    with dpg.texture_registry(show=True):
        dpg.add_static_texture(
            width=608, height=608, default_value=np.zeros((608, 608), dtype=int), tag="texture_tag"
        )
    with dpg.window(
        label="Control",
        modal=False,
        show=True,
        width=300,
        height=1080,
        tag="window-controller",
    ):
        with dpg.group(horizontal=True):
            dpg.add_slider_float(label="ZPos", callback=slider_callback)
        dpg.add_button(label="Exit", callback=exit_callback)

    with dpg.window(
        label="Render",
        modal=False,
        show=True,
        width=608,
        height=608,
        tag="window-render",
    ):
        pass

    dpg.create_viewport(title="Average Enjoyer", width=1080, height=1080)
    dpg.set_viewport_pos(pos=[1260, 0])
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # to enable debugger
    if args.debug_yes:
        while dpg.is_dearpygui_running():
            jobs = dpg.get_callback_queue()
            dpg.run_callbacks(jobs)
            dpg.render_dearpygui_frame()
    else:
        dpg.start_dearpygui()

    dpg.destroy_context()


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

    cvae_dict = torch.load(args.model_path)
    for line in cvae_dict["imports"].split("\n"):
        exec(line)
    exec(cvae_dict["model_src"])
    cvae = eval(cvae_dict["model_cls_name"])(
        *cvae_dict["model_args"], **cvae_dict["model_kwargs"]
    )
    cvae.load_state_dict(cvae_dict["model_state_dict"])
    cvae.requires_grad_(False)
    cvae = cvae.to("cuda")

    global_steps_elapsed = cvae_dict["global_steps_elapsed"]
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
    xs = rb_rot_sixd[selected_valid_idxs].reshape(-1, 24, 6)[:, 1:]
    xs = xs.reshape(xs.shape[0], -1)
    # xs = xs[selected_valid_idxs]
    xs = torch.tensor(xs, dtype=torch.float, device="cuda")
    ys = np.concatenate(
        [
            data["rb_pos"].reshape(-1, 24, 3)[:, 0, -1, None],  # z position
            data["rb_rot_sixd"].reshape(-1, 24, 6)[:, 0],
            data["rb_vel"].reshape(-1, 24, 3)[:, 0],
            data["rb_ang"].reshape(-1, 24, 3)[:, 0],
        ],
        axis=-1,
    )
    ys = ys[selected_valid_idxs]
    ys = torch.tensor(ys, dtype=torch.float, device="cuda")
    # _, _, mu, _ = cvae.forward(
    #     torch.as_tensor(xs, dtype=torch.float, device="cuda"),
    #     torch.as_tensor(ys, dtype=torch.float, device="cuda"),
    #     train_yes=False,
    # )
    # muys = torch.cat([mu, ys], dim=-1)
    # rb_rot_sixd_recon = cvae.decoder(muys)
    # rb_rot_sixd_recon = rb_rot_sixd_recon.cpu().detach().numpy()

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

    visualize_dpg_controller(logger, cvae)

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--debug_yes", action="store_true")
    args = parser.parse_args()

    main()
