import os
import sys

from scipy.spatial.transform import Rotation

from mlexp_utils.dirs import proj_dir
from poselib.poselib.skeleton.skeleton3d import (
    SkeletonState,
    SkeletonTree,
)
from viz.visual_data_pv import XMLVisualDataContainer

sys.path.append(os.path.abspath("./src"))

from argparse import ArgumentParser
import numpy as np
import torch
import pyvista as pv

slider_val = 0


class VisualizerRoutine:
    def __init__(self, update):
        # default parameters
        self.update = update
        self.kwargs = {
            "anim_frame": 0,
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


class Main:
    def __init__(self):
        self.engine = VisualizerRoutine(self.update)

        # run_dir = os.getcwd()
        # out_dir = f"{proj_dir}/out/{args.run_name}/{args.out_name}"
        # os.makedirs(out_dir, exist_ok=True)
        # logdir = f"{proj_dir}/logdir/{args.run_name}/{args.out_name}"
        # os.makedirs(logdir, exist_ok=True)
        # logger = my_logging.get_logger(f"{args.out_name}", logdir)
        # logger.info(f"Starting")

        data_path = os.path.join(proj_dir, "data", "nami", "AMASS", "torchready_v4.pkl")
        d = torch.load(data_path)
        print("hehe")

        self.rb_pos = d["rb_pos"]
        self.rb_rot_sixd = d["rb_rot_sixd"]
        self.rb_rot_sixd_inv = d["rb_rot_sixd_inv"]
        self.rb_pos_inv = d["rb_pos_inv"]

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
        self.end_frame = 199
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
        animation = False

        if animation:

            if self.anim_slider:
                self.anim_slider.GetRepresentation().SetValue(int(self.current_frame))
            t = int(self.current_frame)
        else:
            t = self.engine.get_val("anim_frame")  # int(self.current_frame)

        rb_pos_reshaped = self.rb_pos[t].reshape(24, 3)
        rb_rot_sixd_reshaped = self.rb_rot_sixd[t].reshape(24, 3, 2) * 1
        third_column = np.cross(
            rb_rot_sixd_reshaped[..., 0], rb_rot_sixd_reshaped[..., 1], axis=-1
        )
        rb_rot_rotmat = np.concatenate(
            [rb_rot_sixd_reshaped, third_column[..., None]], axis=-1
        )
        rb_rot_quat = Rotation.from_matrix(rb_rot_rotmat).as_quat()
        sk_state = SkeletonState.from_rotation_and_root_translation(
            self.sk_tree,
            torch.as_tensor(rb_rot_quat, dtype=torch.float),
            torch.as_tensor(rb_pos_reshaped[0], dtype=torch.float),
            is_local=False,
        )
        gt_global_translation = sk_state.global_translation.detach().cpu().numpy()
        gt_global_rotation = sk_state.global_rotation.detach().cpu().numpy()

        rb_pos_inv_reshaped = self.rb_pos_inv[t].reshape(24, 3)
        rb_rot_sixd_inv_reshaped = self.rb_rot_sixd_inv[t].reshape(24, 3, 2) * 1
        third_column = np.cross(
            rb_rot_sixd_inv_reshaped[..., 0], rb_rot_sixd_inv_reshaped[..., 1], axis=-1
        )
        rb_rot_rotmat = np.concatenate(
            [rb_rot_sixd_inv_reshaped, third_column[..., None]], axis=-1
        )
        rb_rot_quat = Rotation.from_matrix(rb_rot_rotmat).as_quat()
        sk_state = SkeletonState.from_rotation_and_root_translation(
            self.sk_tree,
            torch.as_tensor(rb_rot_quat, dtype=torch.float),
            torch.as_tensor(rb_pos_inv_reshaped[0], dtype=torch.float),
            is_local=False,
        )
        inv_global_translation = sk_state.global_translation.detach().cpu().numpy()
        inv_global_rotation = sk_state.global_rotation.detach().cpu().numpy()

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
            pos = inv_global_translation[i] * 1
            inv_rotmat = Rotation.from_quat(inv_global_rotation[i]).as_matrix()
            m[:3, :3] = inv_rotmat
            m[:3, 3] = pos
            recon_actor.user_matrix = m

        self.pl.update()

    def define_sliders(self):
        self.anim_slider = self.pl.add_slider_widget(
            lambda value: self.engine("anim_frame", int(value)),
            [self.start_frame, self.end_frame],
            title="anim frame",
            pointa=(0.025, 0.78),
            pointb=(0.31, 0.78),
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


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--run_name", type=str, required=True)
    # parser.add_argument("--out_name", type=str, required=True)
    # parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--debug_yes", action="store_true")
    args = parser.parse_args()

    if args.debug_yes:
        import pydevd_pycharm

        pydevd_pycharm.settrace(
            "localhost",
            port=12345,
            stdoutToServer=True,
            stderrToServer=True,
            suspend=False,
        )

    Main()
