import os
import sys
import debugpy

# 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
debugpy.listen(5679)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# print("break on this line")

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
from pyvista.utilities import threaded

sys.path.append(os.path.abspath("./src"))
import pauli
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
import dearpygui.dearpygui as dpg
import array

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
            width=608,
            height=608,
            default_value=np.zeros((608, 608), dtype=int),
            tag="texture_tag",
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


slider_val = 0


class VisualizerRoutine:
    def __init__(self, update):
        # default parameters
        self.update = update
        self.kwargs = {
            "dim_0": 0,
            "dim_1": 0,
            "dim_2": 0,
            "dim_3": 0,
            "dim_4": 0,
            "dim_5": 0,
            "dim_6": 0,
            "dim_7": 0,
            "dim_8": 0,
            "dim_9": 0,
            "anim_frame": 0,
            "add_x_velocity": 0,
            "add_y_velocity": 0,
        }

    def __call__(self, param, value):
        print(value)
        self.kwargs[param] = value
        print(f"{param}: {value}")
        self.update()

    def get_val(self, param_name):
        return self.kwargs[param_name]


def test(step):
    print(step)


def find_k(value, loaded_sum_map):
    curr_k = None
    for s, k in loaded_sum_map.items():
        if value >= s:
            curr_k = k
        else:
            return curr_k
    return curr_k


class Main:
    def __init__(self):
        self.loaded_sum_map = torch.load("sum_map.pt")
        self.engine = VisualizerRoutine(self.update)

        cvae_dict = torch.load(args.model_path)
        self.cvae = pauli.load(cvae_dict)
        self.cvae.load_state_dict(cvae_dict["model_state_dict"])
        self.cvae.requires_grad_(False)
        self.cvae.eval()
        self.cvae = self.cvae.to("cuda")

        self.current_k = None

        global_steps_elapsed = cvae_dict["global_steps_elapsed"]
        rr.set_time_sequence("GlobalStepsElapsed", global_steps_elapsed)

        current_path = os.getcwd()
        print("Current Path:", current_path)

        pkl_file_path = f"torchready_v5_dof_circle.pkl"
        data = torch.load(pkl_file_path)
        self.rb_rot_sixd = data["rb_rot_sixd"]

        rb_rot_sixd_reshaped = self.rb_rot_sixd.reshape(-1, 24, 3, 2)
        third_column = np.cross(
            rb_rot_sixd_reshaped[..., 0], rb_rot_sixd_reshaped[..., 1], axis=-1
        )
        rb_rot_rotmat = np.concatenate(
            [rb_rot_sixd_reshaped, third_column[..., None]], axis=-1
        )

        self.rb_rot_euler = Rotation.from_matrix(rb_rot_rotmat.reshape(-1, 3, 3)).as_euler(
            "zyx"
        )
        self.cached_root_yaw = self.rb_rot_euler.reshape(
            *rb_rot_sixd_reshaped.shape[:-2], 3
        )[:, [0], 0]

        self.rb_rot_sixd_inv = data["rb_rot_sixd_inv"]
        self.dof_pos = data["dof_pos"]

        self.rb_ang_vel = data["rb_ang"].reshape(-1, 24, 3)
        self.rb_root_ang_vel = self.rb_ang_vel[..., 0, :]

        self.rb_pos = data["rb_pos"]
        self.rb_root_vel_inv = data["rb_vel_inv"].reshape(-1, 24, 3)[..., 0, :]

        self.root_rot = data["root_rot"]

        # n_train = int(self.rb_pos.shape[0] * 0.90)
        # n_valid = self.rb_pos.shape[0] - n_train

        # np.random.seed(0)
        # torch.random.manual_seed(0)
        # torch.cuda.manual_seed_all(0)
        # train_idxs = np.random.choice(self.rb_pos.shape[0], n_train, replace=False)
        # valid_idxs = np.setdiff1d(np.arange(self.rb_pos.shape[0]), train_idxs)

        # selected_train_idxs = np.random.choice(train_idxs, 10, replace=False)
        self.selected_valid_idxs = np.arange(
            self.rb_pos.shape[0]
        )  # np.random.choice(valid_idxs, 10, replace=False)

        # _, rb_rot_sixd_recon, _, _ = vae.forward(
        #     torch.as_tensor(rb_rot_sixd[:10], dtype=torch.float, device="cuda"),
        #     train_yes=False,
        # )
        # self.xs = rb_rot_sixd[self.selected_valid_idxs].reshape(-1, 24, 6)[:, 1:]
        self.xs =  self.dof_pos

        self.ys_lower = self.rb_root_vel_inv * 1

        self.xs = self.xs.reshape(self.xs.shape[0],23, -1)
        # self.xs = self.xs[self.selected_valid_idxs]
        self.xs = torch.tensor(self.xs, dtype=torch.float, device="cuda")
        self.ys_lower = torch.tensor(self.ys_lower, dtype=torch.float, device="cuda")
        
        # _, _, mu, _ = cvae.forward(
        #     torch.as_tensor(self.xs, dtype=torch.float, device="cuda"),
        #     torch.as_tensor(self.ys, dtype=torch.float, device="cuda"),
        #     train_yes=False,
        # )
        # muys = torch.cat([mu, self.ys], dim=-1)
        # rb_rot_sixd_recon = cvae.decoder(muys)
        # rb_rot_sixd_recon = rb_rot_sixd_recon.cpu().detach().numpy()

        # Reconstruction visual dumping
        self.sk_tree = SkeletonTree.from_mjcf(
            "phc/data/assets/mjcf/smpl_humanoid_1.xml"
        )
        gt_visual_data = XMLVisualDataContainer(
            "phc/data/assets/mjcf/my_smpl_humanoid.xml"
        )
        recon_visual_data = XMLVisualDataContainer(
            "phc/data/assets/mjcf/my_smpl_humanoid.xml"
        )
        # self.pl = pv.Plotter(off_screen=True, window_size=(608, 608))
        self.pl = pv.Plotter(off_screen=False, window_size=(608, 608))
        self.pl.enable_shadows()
        # self.pl.add_mesh(gt_visual_data.plane)
        self.pl.add_mesh(
            pv.Cube(center=(0, 0, -0.5), x_length=100, y_length=100), color="#237a3c"
        )
        self.pl.add_axes()
        distance = 5
        self.pl.camera.position = (-distance, -distance, 4)
        self.pl.camera.focal_point = (0, 0, 0)

        self.gt_actors = []
        for mesh, ax in zip(gt_visual_data.meshes, gt_visual_data.axes):
            actor = self.pl.add_mesh(mesh, color="blue")
            self.gt_actors.append(actor)

        self.recon_actors = []
        for mesh, ax in zip(recon_visual_data.meshes, recon_visual_data.axes):
            actor = self.pl.add_mesh(mesh, color="red")
            self.recon_actors.append(actor)

        self.start_frame = 0
        self.end_frame = 800
        self.current_frame = self.start_frame
        self.speed = 1
        self.anim_slider = None
        self.pl.show(interactive_update=True, auto_close=False, interactive=True)

        self.define_sliders()
        while True:
            self.update()

    def update(self):
        # print(delta_time)
        self.current_frame += self.speed
        if self.current_frame >= self.end_frame:
            self.current_frame = self.start_frame
        animation = True
        if animation == True:

            if self.anim_slider:
                self.anim_slider.GetRepresentation().SetValue(int(self.current_frame))
            t = int(self.current_frame)
        else:
            t = self.engine.get_val("anim_frame")  # int(self.current_frame)

        new_k = find_k(t, self.loaded_sum_map)
        if self.current_k != new_k:
            self.current_k = new_k
            #print(self.current_k)
        rb_pos_reshaped = self.rb_pos[self.selected_valid_idxs[t]].reshape(24, 3)
        rb_rot_sixd_reshaped = (
            self.rb_rot_sixd[self.selected_valid_idxs[t]].reshape(24, 3, 2) * 1
        )
        # rb_rot_sixd_reshaped = self.sixd_add_root(rb_rot_sixd_reshaped,torch.tensor((self.cached_root_yaw[t][0],0,0)).numpy()).reshape(24, 3, 2)

        third_column = np.cross(
            rb_rot_sixd_reshaped[..., 0], rb_rot_sixd_reshaped[..., 1], axis=-1
        )
        rb_rot_rotmat = np.concatenate(
            [rb_rot_sixd_reshaped, third_column[..., None]], axis=-1
        )
        # rb_rot = Rotation.from_matrix(rb_rot_rotmat)
        # correction = Rotation.from_euler("Z", -1 * self.cached_root_yaw[t][0])
        # rb_rot = correction * rb_rot
        # rb_rot_quat = rb_rot.as_quat()
        rb_rot_quat = Rotation.from_matrix(rb_rot_rotmat).as_quat()
        #print(np.linalg.norm(rb_pos_reshaped[0, :2]))
        rb_pos_reshaped[0, :2] *= 0
        sk_state = SkeletonState.from_rotation_and_root_translation(
            self.sk_tree,
            torch.as_tensor(rb_rot_quat, dtype=torch.float),
            torch.as_tensor(rb_pos_reshaped[0], dtype=torch.float),
            is_local=False,
        )
        gt_global_translation = sk_state.global_translation.detach().cpu().numpy()
        gt_global_rotation = sk_state.global_rotation.detach().cpu().numpy()

        orig_x = self.xs[[t]]
        orid_y_lower = self.ys_lower[[t]]
        print(orid_y_lower)
        #orig_y = self.ys[[t]]
        # i = 1
        # if(i==1):
        #     torch.save(orig_x, 'xs.pkl')
        #     torch.save(orig_y, 'ys.pkl')
        #     i+=1
        add_x_vel = self.engine.get_val("add_x_velocity")
        add_y_vel = self.engine.get_val("add_y_velocity")

        _, decoded, mu, log_var = self.cvae.forward(
            orig_x,
            orid_y_lower,
            train_yes=False,
        )
        z = mu

        orid_y_lower[...,0] += add_x_vel * 3
        orid_y_lower[...,1] += add_y_vel * 3
        # orig_y[..., :] *= 0
        # print(orig_y[..., 7:])
        # print(engine.get_val('dim_0'))
        z[..., -10] += self.engine.get_val("dim_0")
        z[..., -11] += self.engine.get_val("dim_1")
        z[..., -12] += self.engine.get_val("dim_2")
        z[..., -13] += self.engine.get_val("dim_3")
        z[..., -14] += self.engine.get_val("dim_4")
        if z.shape[-1] > 5:
            z[..., -5] += self.engine.get_val("dim_5")
            z[..., -6] += self.engine.get_val("dim_6")
            z[..., -7] += self.engine.get_val("dim_7")
            z[..., -8] += self.engine.get_val("dim_8")
            z[..., -9] += self.engine.get_val("dim_9")

        # cvae_decoded = self.cvae.decode(z, orig_y)
        #z = z.reshape(z.shape[0], 14)
        #hammer_sizes = [6,10]
        #hammer_sizes = 
        current_start = 0
        decodeds=[]
        for index, hs in enumerate(self.cvae.hammer_size):
            print(current_start)
            print(current_start + hs)
            if(index == 0):
                ys = orid_y_lower
            else:
                ys = decodeds[0]
            chain_z = z[:, current_start: current_start + hs] * 1
            current_start += hs
            decodeds.append(self.cvae.decode(chain_z,ys, index))
        cvae_decoded = torch.concatenate(decodeds, axis=-1)
        cvae_decoded = cvae_decoded.cpu().reshape((1, 23, -1))

        pose_quat = Rotation.from_rotvec(cvae_decoded.reshape(-1, 3)).as_quat().reshape(-1, 23, 4)
        root_rot = self.root_rot[self.selected_valid_idxs[t]]

        pose_quat = np.concatenate([root_rot[None, None, :], pose_quat], axis=1)
        
        mutated_trans = rb_pos_reshaped[0] * 1
        recon_sk_state = SkeletonState.from_rotation_and_root_translation(
            self.sk_tree,
            torch.as_tensor(pose_quat,dtype=torch.float).cpu(),
            torch.as_tensor(mutated_trans,dtype=torch.float).cpu(),
            is_local=True
        )
        recon_global_translation = (
            recon_sk_state.global_translation.detach().cpu().numpy()
        )
        recon_global_rotation = (
            recon_sk_state.global_rotation.detach().cpu().numpy()
        )

        for i in range(len(self.gt_actors)):
            gt_actor = self.gt_actors[i]
            m = np.eye(4)
            pos = gt_global_translation[i] * 1
            gt_rotmat = Rotation.from_quat(gt_global_rotation[i]).as_matrix()
            m[:3, :3] = gt_rotmat
            m[:3, 3] = pos
            gt_actor.user_matrix = m

            recon_actor = self.recon_actors[i]
            m = np.eye(4)
            recon_rotmat = Rotation.from_quat(recon_global_rotation[0,i]).as_matrix()
            # pos = cvae_decoded[i] * 1
            m[:3, :3] = recon_rotmat
            m[:3, 3] = recon_global_translation[0, i] * 1
            recon_actor.user_matrix = m

        # self.pl.render()

        # img = self.pl.screenshot()
        # plt.figure()
        # plt.imshow(img)
        # plt.show()

        self.pl.update()
        # self.pl.show(interactive_update=True, auto_close=False, interactive=True)

    def define_sliders(self):
        start_val = -3
        end_val = 3
        self.pl.add_slider_widget(
            lambda value: self.engine("dim_0", (value)),
            [start_val, end_val],
            title="0 dim",
            pointa=(0.025, 0.1),
            pointb=(0.31, 0.1),
            interaction_event="always",
            style="modern",
        )

        self.pl.add_slider_widget(
            lambda value: self.engine("dim_1", (value)),
            [start_val, end_val],
            title="1 dim",
            pointa=(0.35, 0.1),
            pointb=(0.64, 0.1),
            interaction_event="always",
            style="modern",
        )

        self.pl.add_slider_widget(
            lambda value: self.engine("dim_2", (value)),
            [start_val, end_val],
            title="2 dim",
            pointa=(0.67, 0.1),
            pointb=(0.98, 0.1),
            interaction_event="always",
            style="modern",
        )

        self.pl.add_slider_widget(
            lambda value: self.engine("dim_3", (value)),
            [start_val, end_val],
            title="3 dim",
            pointa=(0.025, 0.23),
            pointb=(0.31, 0.23),
            interaction_event="always",
            style="modern",
        )

        self.pl.add_slider_widget(
            lambda value: self.engine("dim_4", (value)),
            [start_val, end_val],
            title="4 dim",
            pointa=(0.35, 0.23),
            pointb=(0.64, 0.23),
            interaction_event="always",
            style="modern",
        )

        self.pl.add_slider_widget(
            lambda value: self.engine("dim_5", (value)),
            [start_val, end_val],
            title="5 dim",
            pointa=(0.67, 0.23),
            pointb=(0.98, 0.23),
            interaction_event="always",
            style="modern",
        )

        self.pl.add_slider_widget(
            lambda value: self.engine("dim_6", (value)),
            [start_val, end_val],
            title="6 dim",
            pointa=(0.025, 0.35),
            pointb=(0.31, 0.35),
            interaction_event="always",
            style="modern",
        )

        self.pl.add_slider_widget(
            lambda value: self.engine("dim_7", (value)),
            [start_val, end_val],
            title="7 dim",
            pointa=(0.35, 0.35),
            pointb=(0.64, 0.35),
            interaction_event="always",
            style="modern",
        )

        self.pl.add_slider_widget(
            lambda value: self.engine("dim_8", (value)),
            [start_val, end_val],
            title="8 dim",
            pointa=(0.67, 0.35),
            pointb=(0.98, 0.35),
            interaction_event="always",
            style="modern",
        )

        self.pl.add_slider_widget(
            lambda value: self.engine("dim_9", (value)),
            [start_val, end_val],
            title="9 dim",
            pointa=(0.025, 0.9),
            pointb=(0.31, 0.9),
            interaction_event="always",
            style="modern",
        )

        self.anim_slider = self.pl.add_slider_widget(
            lambda value: self.engine("anim_frame", int(value)),
            [self.start_frame, self.end_frame],
            title="anim frame",
            pointa=(0.025, 0.78),
            pointb=(0.31, 0.78),
            interaction_event="always",
            style="modern",
        )

        self.pl.add_slider_widget(
            lambda value: self.engine("add_x_velocity", (value)),
            [start_val, end_val],
            title="add x velocity",
            pointa=(0.025, 0.66),
            pointb=(0.31, 0.66),
            interaction_event="always",
            style="modern",
        )

        self.pl.add_slider_widget(
            lambda value: self.engine("add_y_velocity", (value)),
            [start_val, end_val],
            title="add y velocity",
            pointa=(0.025, 0.54),
            pointb=(0.31, 0.54),
            interaction_event="always",
            style="modern",
        )

    
    def sixd_add_root(self, rb_rot_sixd, orientation):
        rb_rot_sixd_reshaped = rb_rot_sixd.reshape(-1, 24, 3, 2)
        third_column = np.cross(
            rb_rot_sixd_reshaped[..., 0], rb_rot_sixd_reshaped[..., 1], axis=-1
        )
        rb_rot_rotmat = np.concatenate(
            [rb_rot_sixd_reshaped, third_column[..., None]], axis=-1
        )
        rb_rot_euler = Rotation.from_matrix(rb_rot_rotmat.reshape(-1, 3, 3)).as_euler(
            "zyx"
        )

        rb_rot_euler.reshape(*rb_rot_sixd_reshaped.shape[:-2], 3)[:] += orientation
        new_rb_rot_sixd = (
            Rotation.from_euler("zyx", rb_rot_euler)
            .as_matrix()
            .reshape((*rb_rot_sixd_reshaped.shape[:-2], 3, 3))[..., :2]
            .reshape((*rb_rot_sixd_reshaped.shape[:-2], 6))
        )
        return new_rb_rot_sixd


def draw_image():
    dpg.create_context()

    texture_data = []
    for i in range(0, 100 * 100):
        texture_data.append(255 / 255)
        texture_data.append(0)
        texture_data.append(255 / 255)
        texture_data.append(255 / 255)

    raw_data = array.array("f", texture_data)

    with dpg.texture_registry(show=True):
        dpg.add_raw_texture(
            width=100,
            height=100,
            default_value=raw_data,
            format=dpg.mvFormat_Float_rgba,
            tag="texture_tag",
        )

    def update_dynamic_texture(sender, app_data, user_data):
        new_color = dpg.get_value(sender)
        new_color[0] = new_color[0] / 255
        new_color[1] = new_color[1] / 255
        new_color[2] = new_color[2] / 255
        new_color[3] = new_color[3] / 255

        for i in range(0, 100 * 100 * 4):
            raw_data[i] = new_color[i % 4]

    with dpg.window(label="Tutorial"):
        dpg.add_image("texture_tag")
        dpg.add_color_picker(
            (255, 0, 255, 255),
            label="Texture",
            no_side_preview=True,
            alpha_bar=True,
            width=200,
            callback=update_dynamic_texture,
        )

    dpg.create_viewport(title="Custom Title", width=800, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--run_name", type=str, required=True)
#     parser.add_argument("--out_name", type=str, required=True)
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--debug_yes", action="store_true")
#     args = parser.parse_args()

#     Main()
parser = ArgumentParser()
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--out_name", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--debug_yes", action="store_true")
args = parser.parse_args()

Main()
