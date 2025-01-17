import time
import numpy as np
from rl_games.algos_torch import players
import torch
import vispy
from isaacgymenvs.utils.visual_data import VisualDataContainer
from vispy import scene
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import io
import os

# from xvfbwrapper import Xvfb

# vdisplay = Xvfb()
# vdisplay.start()


class MyPlayer(players.PpoPlayerContinuous):
    pass
    # def get_distance_for_body(self, body_pos, body_name):
    #     green_guy_pos = body_pos[:, 0, self.env._body_names.index(body_name), :]
    #     red_guy_pos = body_pos[:, 1, self.env._body_names.index(body_name), :]
    #     min_dist = torch.cdist(green_guy_pos, red_guy_pos).min()
    #     return min_dist
    #
    # def run(self):
    #     self.train_dir = self.env.cfg.get("train_dir", "runs")
    #
    #     checkpoint_path = self.env.cfg["checkpoint"]
    #     parts = checkpoint_path.split("/")
    #     index_runs = parts.index("runs")
    #     self.experiment_name = parts[index_runs + 1]
    #
    #     d = torch.load(checkpoint_path)
    #     self.last_checkpoint = d["epoch"]
    #     self.experiment_dir = os.path.join(
    #         self.train_dir, self.experiment_name + "_Eval"
    #     )
    #
    #     self.summaries_dir = os.path.join(self.experiment_dir, "summaries")
    #     print(self.summaries_dir)
    #
    #     self.writer = SummaryWriter(self.summaries_dir)
    #
    #     n_games = self.games_num
    #     render = self.render_env
    #     n_game_life = self.n_game_life
    #     is_determenistic = self.is_determenistic
    #     sum_rewards = 0
    #     sum_steps = 0
    #     sum_game_res = 0
    #     n_games = n_games * n_game_life
    #     games_played = 0
    #     has_masks = False
    #     has_masks_func = getattr(self.env, "has_action_mask", None) is not None
    #     vispy.use("egl")
    #     canvas = scene.SceneCanvas(size=(800, 600), show=False)
    #     view = canvas.central_widget.add_view()
    #     view.camera = scene.TurntableCamera(
    #         up="z", fov=1.0, elevation=15.0, distance=300, azimuth=15
    #     )
    #     view.camera.rect = 0, 0, 1, 1
    #     _ = scene.visuals.XYZAxis(parent=view.scene)
    #     vd = VisualDataContainer(view)
    #     img = canvas.render()
    #
    #     op_agent = getattr(self.env, "create_agent", None)
    #     if op_agent:
    #         agent_inited = True
    #         # print('setting agent weights for selfplay')
    #         # self.env.create_agent(self.env.config)
    #         # self.env.set_weights(range(8),self.get_weights())
    #
    #     if has_masks_func:
    #         has_masks = self.env.has_action_mask()
    #
    #     need_init_rnn = self.is_rnn
    #     for _ in range(n_games):
    #         if games_played >= n_games:
    #             break
    #
    #         obses = self.env_reset(self.env)
    #         batch_size = 1
    #         batch_size = self.get_batch_size(obses, batch_size)
    #
    #         if need_init_rnn:
    #             self.init_rnn()
    #             need_init_rnn = False
    #
    #         cr = torch.zeros(batch_size, dtype=torch.float32)
    #         steps = torch.zeros(batch_size, dtype=torch.float32)
    #
    #         print_game_res = False
    #         images = []
    #         rotating_images = [[] for _ in range(8)]
    #         single_rotating_images = [[] for _ in range(8)]
    #         self.steps_array = []
    #         min_dist_list = []
    #         anim_min_dist_list = []
    #         self.define_history_list()
    #
    #         for n in range(self.max_steps):
    #             if has_masks:
    #                 masks = self.env.get_action_mask()
    #                 action = self.get_masked_action(obses, masks, is_determenistic)
    #             else:
    #                 action = self.get_action(obses, is_determenistic)
    #             obses, r, done, info = self.env_step(self.env, action)
    #             cr += r
    #             steps += 1
    #
    #             self.steps_array.append(steps[0].item())
    #
    #             if self.env.num_envs == 1:
    #
    #                 ref_anim_pos = (
    #                     self.env.vid_visualized_rb_body_pos[0].detach().cpu().numpy()
    #                 )
    #
    #                 temp_rigid_body_pos = self.env._rigid_body_pos.clone().reshape(
    #                     1, 48, 3
    #                 )
    #                 temp_rigid_body_pos[..., :2] -= self.env.additive_agent_pos[..., :2]
    #                 temp_rigid_body_pos = temp_rigid_body_pos.reshape(1, 2, -1, 3)
    #                 temp_rigid_body_pos[:, 1, ...] += (
    #                     self.env.additive_agent_pos.reshape(1, 2, -1, 3)[:, 1, ...]
    #                     - self.env.additive_agent_pos.reshape(1, 2, -1, 3)[:, 0, ...]
    #                 )
    #
    #                 temp_rigid_body_pos = (
    #                     temp_rigid_body_pos.reshape(1, 48, 3)[0].cpu().numpy()
    #                 )
    #
    #                 vd.body_markers[0].set_data(
    #                     pos=temp_rigid_body_pos,
    #                     size=0.1,
    #                     edge_width=0.0,
    #                     edge_color=(0, 0, 0, 0),
    #                     face_color=(1, 0, 0, 1),
    #                 )
    #
    #                 vd.body_markers[1].set_data(
    #                     pos=np.array([[0, 0, 0]]), size=0.1, edge_width=0.01
    #                 )
    #                 # for i in [0,45,90,135,180,225,270,315]:
    #                 #     vd.view.camera.azimuth = i
    #                 #     vd.view.camera.elevation = 45
    #                 #     canvas.update()
    #                 #     single_rotating_images[i//45].append(canvas.render())
    #
    #                 vd.body_markers[1].set_data(
    #                     pos=ref_anim_pos,
    #                     size=0.1,
    #                     edge_width=0.0,
    #                     edge_color=(0, 0, 0, 0),
    #                     face_color=(0, 0, 1, 1),
    #                 )
    #
    #                 vd.view.camera.azimuth = 0
    #                 vd.view.camera.elevation = 90
    #                 canvas.update()
    #                 img0 = canvas.render()
    #
    #                 # vd.view.camera.azimuth = 0
    #                 # vd.view.camera.elevation = 0
    #                 # canvas.update()
    #                 # img1 = canvas.render()
    #
    #                 # vd.view.camera.azimuth = 90
    #                 # vd.view.camera.elevation = 0
    #                 # canvas.update()
    #                 # img2 = canvas.render()
    #
    #                 # vd.view.camera.azimuth = 180
    #                 # vd.view.camera.elevation = 0
    #                 # canvas.update()
    #                 # img3 = canvas.render()
    #
    #                 # vd.view.camera.azimuth = -90
    #                 # vd.view.camera.elevation = 0
    #                 # canvas.update()
    #                 # img4 = canvas.render()
    #
    #                 vd.view.camera.azimuth = 90
    #                 vd.view.camera.elevation = 45
    #                 canvas.update()
    #                 img5 = canvas.render()
    #
    #                 # big_img = np.concatenate([img0,img5, img1,img2,img3,img4], axis=1)
    #                 big_img = np.concatenate([img0, img5], axis=1)
    #                 images.append(big_img)
    #
    #                 # import matplotlib.pyplot as plt
    #                 # plt.figure()
    #                 # plt.imshow(big_img)
    #                 # plt.pause(0.1)
    #                 # plt.show()
    #
    #                 # for i in [0,45,90,135,180,225,270,315]:
    #                 #     vd.view.camera.azimuth = i
    #                 #     vd.view.camera.elevation = 45
    #                 #     canvas.update()
    #                 #     rotating_images[i//45].append(canvas.render())
    #
    #             if render:
    #                 self.env.render(mode="human")
    #                 time.sleep(self.render_sleep)
    #
    #             if self.env.num_envs == 1:
    #                 green_guy_pos = self.env._rigid_body_pos[:, 0, ...]
    #                 red_guy_pos = self.env._rigid_body_pos[:, 1, ...]
    #                 min_dist = torch.cdist(green_guy_pos, red_guy_pos).min()
    #
    #                 anim_min_dist_list.append(
    #                     torch.cdist(
    #                         self.env.modified_rb_body_pos.reshape(1, 2, 24, 3)[
    #                             :, 0, ...
    #                         ],
    #                         self.env.modified_rb_body_pos.reshape(1, 2, 24, 3)[
    #                             :, 1, ...
    #                         ],
    #                     ).min()
    #                 )
    #                 min_dist_list.append(min_dist)
    #                 self.update_history_lists()
    #
    #             all_done_indices = done.nonzero(as_tuple=False)
    #             done_indices = all_done_indices[:: self.num_agents]
    #             done_count = len(done_indices)
    #             games_played += done_count
    #
    #             if done_count > 0:
    #                 if self.is_rnn:
    #                     for s in self.states:
    #                         s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0
    #
    #                 cur_rewards = cr[done_indices].sum().item()
    #                 cur_steps = steps[done_indices].sum().item()
    #
    #                 cr = cr * (1.0 - done.float())
    #                 steps = steps * (1.0 - done.float())
    #                 sum_rewards += cur_rewards
    #                 sum_steps += cur_steps
    #                 print(sum_rewards)
    #
    #                 game_res = 0.0
    #                 if isinstance(info, dict):
    #                     if "battle_won" in info:
    #                         print_game_res = True
    #                         game_res = info.get("battle_won", 0.5)
    #                     if "scores" in info:
    #                         print_game_res = True
    #                         game_res = info.get("scores", 0.5)
    #
    #                 if self.print_stats:
    #                     if print_game_res:
    #                         print(
    #                             "reward:",
    #                             cur_rewards / done_count,
    #                             "steps:",
    #                             cur_steps / done_count,
    #                             "w:",
    #                             game_res,
    #                         )
    #                     else:
    #
    #                         # TODO all videos
    #                         if self.env.num_envs == 1:
    #
    #                             # for i in range(8):
    #                             #     vid = np.stack(rotating_images[i])[None]
    #                             #     self.writer.add_video(f'Multi_Video/{i*45}', vid, self.last_checkpoint, dataformats="NTHWC", fps=30)
    #                             #     print(f'video{i} added')
    #                             # rotating_images = [[] for _ in range(8)]
    #
    #                             # for i in range(8):
    #                             #     vid = np.stack(single_rotating_images[i])[None]
    #                             #     self.writer.add_video(f'Single_Video/{i*45}', vid, self.last_checkpoint, dataformats="NTHWC", fps=30)
    #                             #     print(f'single video{i} added')
    #                             # single_rotating_images = [[] for _ in range(8)]
    #
    #                             nvid = np.stack(images)[None]
    #                             # TODO uncomment this
    #                             self.writer.add_video(
    #                                 "Video/v",
    #                                 nvid,
    #                                 self.last_checkpoint,
    #                                 dataformats="NTHWC",
    #                                 fps=25,
    #                             )
    #                             print("video added")
    #
    #                             plt.plot(
    #                                 self.steps_array, min_dist_list, color="red"
    #                             )  # Plot the values against the number of epochs
    #                             plt.plot(
    #                                 self.steps_array, anim_min_dist_list, color="blue"
    #                             )
    #                             plt.xlabel("steps")
    #                             plt.ylabel("Min distance")
    #                             plt.grid(True)
    #                             self.writer.add_figure(
    #                                 "Test/Min distance rigid body",
    #                                 plt.gcf(),
    #                                 global_step=self.last_checkpoint,
    #                             )
    #
    #                             self.plot_info()
    #
    #                         self.steps_array = []
    #                         images = []
    #
    #                         print(
    #                             "reward:",
    #                             cur_rewards / done_count,
    #                             "steps:",
    #                             cur_steps / done_count,
    #                         )
    #
    #                 sum_game_res += game_res
    #                 if batch_size // self.num_agents == 1 or games_played >= n_games:
    #                     break
    #         # # import imageio
    #         # # #imageio.imsave(f"{int(steps[0].item()):03d}.png", big_img)
    #         # # #imageio.mimsave(f"videos/{int(_)}.gif", images[:100])
    #         # # writer =imageio.get_writer(f"videos/{int(_)}.mp4", fps=30)
    #         # # for im in images:
    #         # #     writer.append_data((im))
    #         # # writer.close()
    #         # # print('Saved video')
    #
    #     print(sum_rewards)
    #     print(self.last_checkpoint)
    #     self.writer.add_scalar(
    #         f"Test/Average Reward",
    #         sum_rewards / games_played * n_game_life,
    #         self.last_checkpoint,
    #     )
    #     self.writer.add_scalar(
    #         f"Test/Average Steps",
    #         sum_steps / games_played * n_game_life,
    #         self.last_checkpoint,
    #     )
    #
    #     if print_game_res:
    #         print(
    #             "av reward:",
    #             sum_rewards / games_played * n_game_life,
    #             "av steps:",
    #             sum_steps / games_played * n_game_life,
    #             "winrate:",
    #             sum_game_res / games_played * n_game_life,
    #         )
    #     else:
    #         print(
    #             "av reward:",
    #             sum_rewards / games_played * n_game_life,
    #             "av steps:",
    #             sum_steps / games_played * n_game_life,
    #         )
    #
    # # dof_names = [ 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'L_Toe_1', 'L_Toe_1_1', 'L_Toe_2', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'R_Toe_1', 'R_Toe_1_1', 'R_Toe_2', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
    # #        'R_Elbow', 'R_Wrist', 'R_Hand']
    # def define_history_list(self):
    #
    #     self.rigid_body_history = []
    #     self.ref_anim_history = []
    #     self.contact_forces_history = []
    #
    # def update_history_lists(self):
    #
    #     self.rigid_body_history.append(self.env._rigid_body_pos.clone())
    #     self.ref_anim_history.append(
    #         self.env.modified_rb_body_pos.reshape(1, 2, 24, 3).clone()
    #     )
    #
    #     reshaped_contact_forces = self.env._contact_forces.reshape(1, 2, 24, 3)
    #     self.contact_forces_history.append(reshaped_contact_forces.clone())
    #
    # def plot_info(self):
    #
    #     for body_name in self.env._body_names:
    #         self.plot_dist(body_name)
    #
    #     for body_name in self.env._body_names:
    #         self.plot_contact_forces(body_name)
    #
    #     for body_name in self.env._body_names:
    #         self.plot_cumulative_contact_forces(body_name)
    #
    # def get_distance_from_history(self, body_name, history):
    #     return [self.get_distance_for_body(tensor, body_name) for tensor in history]
    #
    # def get_contact_force_from_history(self, agent, body_name):
    #     # index 1 => Red guy :) Moving character
    #     return [
    #         torch.norm(tensor[:, agent, self.env._body_names.index(body_name), :])
    #         for tensor in self.contact_forces_history
    #     ]
    #
    # def plot_cumulative_contact_forces(self, body_name):
    #     self.writer.add_scalar(
    #         f"ComulativeContactForce/{body_name}",
    #         torch.stack(self.get_contact_force_from_history(1, body_name)).sum(),
    #         self.last_checkpoint,
    #     )
    #
    # def plot_contact_forces(self, body_name):
    #     plt.plot(
    #         self.steps_array,
    #         self.get_contact_force_from_history(0, body_name),
    #         color="blue",
    #     )
    #     plt.plot(
    #         self.steps_array,
    #         self.get_contact_force_from_history(1, body_name),
    #         color="red",
    #     )  # Plot the values against the number of epochs
    #     plt.xlabel("steps")
    #     plt.ylabel("force magnitude")
    #     plt.grid(True)
    #     self.writer.add_figure(
    #         f"ContactForce/{body_name} force magnitude",
    #         plt.gcf(),
    #         global_step=self.last_checkpoint,
    #     )
    #
    # def plot_dist(self, body_name):
    #     plt.plot(
    #         self.steps_array,
    #         self.get_distance_from_history(body_name, self.rigid_body_history),
    #         color="red",
    #     )  # Plot the values against the number of epochs
    #     plt.plot(
    #         self.steps_array,
    #         self.get_distance_from_history(body_name, self.ref_anim_history),
    #         color="blue",
    #     )
    #     plt.xlabel("steps")
    #     plt.ylabel("Min distance")
    #     plt.grid(True)
    #     self.writer.add_figure(
    #         f"Distance/{body_name} min distance",
    #         plt.gcf(),
    #         global_step=self.last_checkpoint,
    #     )


# vdisplay.stop()
