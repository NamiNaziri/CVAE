import os
import time

import numpy as np
import torch
from tqdm import tqdm

from mlexp_utils import my_logging
from mlexp_utils.dirs import proj_dir
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent
import pyvista as pv


class MyAgent(A2CAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        log_dir = os.path.join(
            proj_dir, "logdir", self.config["run_name"], self.config["out_name"]
        )
        self.logger = my_logging.get_logger(self.config["out_name"], log_dir)
        self.logger.info("MyAgent initialized")
        self.model.a2c_network.sigma.requires_grad = False

        self.pl = pv.Plotter(off_screen=True, window_size=(608, 608))
        self.pl.enable_shadows()
        self.pl.add_mesh(
            pv.Cube(center=(0, 0, -0.5), x_length=100, y_length=100), color="white"
        )
        self.pl.add_axes()
        distance = 5
        self.pl.camera.position = (-distance, -distance, 4)
        self.pl.camera.focal_point = (0, 0, 0)

        self.blue_actors = []
        self.red_actors = []
        for _ in range(24):
            sphere = pv.Sphere(radius=0.1)
            blue_actor = self.pl.add_mesh(sphere, color="blue")
            red_actor = self.pl.add_mesh(sphere, color="red")
            self.blue_actors.append(blue_actor)
            self.red_actors.append(red_actor)

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            self.hvd.setup_algo(self)

        while True:
            epoch_num = self.update_epoch()
            (
                step_time,
                play_time,
                update_time,
                sum_time,
                a_losses,
                c_losses,
                b_losses,
                entropies,
                kls,
                last_lr,
                lr_mul,
            ) = self.train_epoch()
            self.model.a2c_network.sigma.data = torch.clip(
                self.model.a2c_network.sigma - 0.01, -5.0, 0.5
            )

            total_time += sum_time
            frame = self.frame

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            if self.multi_gpu:
                self.hvd.sync_stats(self)
            should_exit = False
            if self.rank == 0:
                # do we need scaled_time?
                scaled_time = sum_time  # self.num_agents * sum_time
                scaled_play_time = play_time  # self.num_agents * play_time
                curr_frames = self.curr_frames
                self.frame += curr_frames
                if self.print_stats:
                    fps_step = curr_frames / step_time
                    fps_step_inference = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    self.logger.info(
                        f"fps step: {fps_step:.1f} fps step and policy inference: {fps_step_inference:.1f}  fps total: {fps_total:.1f}"
                    )

                self.write_stats(
                    total_time,
                    epoch_num,
                    step_time,
                    play_time,
                    update_time,
                    a_losses,
                    c_losses,
                    entropies,
                    kls,
                    last_lr,
                    lr_mul,
                    frame,
                    scaled_time,
                    scaled_play_time,
                    curr_frames,
                )
                self.writer.add_scalar(
                    "info/logsigma",
                    torch.mean(self.model.a2c_network.sigma).item(),
                    frame,
                )
                if len(b_losses) > 0:
                    self.writer.add_scalar(
                        "losses/bounds_loss",
                        torch_ext.mean_list(b_losses).item(),
                        frame,
                    )

                if self.has_soft_aug:
                    self.writer.add_scalar(
                        "losses/aug_loss", np.mean(aug_losses), frame
                    )

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = "rewards" if i == 0 else "rewards{0}".format(i)
                        self.writer.add_scalar(
                            rewards_name + "/step".format(i), mean_rewards[i], frame
                        )
                        self.writer.add_scalar(
                            rewards_name + "/iter".format(i), mean_rewards[i], epoch_num
                        )
                        self.writer.add_scalar(
                            rewards_name + "/time".format(i),
                            mean_rewards[i],
                            total_time,
                        )

                    self.writer.add_scalar("episode_lengths/step", mean_lengths, frame)
                    self.writer.add_scalar(
                        "episode_lengths/iter", mean_lengths, epoch_num
                    )
                    self.writer.add_scalar(
                        "episode_lengths/time", mean_lengths, total_time
                    )

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = (
                        self.config["name"]
                        + "_ep_"
                        + str(epoch_num)
                        + "_rew_"
                        + str(mean_rewards[0])
                    )

                    # if self.save_freq > 0:
                    #     if (epoch_num % self.save_freq == 0) and (
                    #         mean_rewards[0] <= self.last_mean_rewards
                    #     ):
                    #         self.save(
                    #             os.path.join(self.nn_dir, "last_" + checkpoint_name)
                    #         )
                    #         self.logger.info(
                    #             "saving last rewards: ", mean_rewards, " epoch: ", epoch_num
                    #         )

                    if (
                        mean_rewards[0] > self.last_mean_rewards
                        and epoch_num >= self.save_best_after
                    ):
                        self.logger.info(f"Saving next best rewards: {mean_rewards}")
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config["name"]))
                        if self.last_mean_rewards > self.config["score_to_win"]:
                            self.logger.info("Network won!")
                            self.save(os.path.join(self.nn_dir, checkpoint_name))
                            should_exit = True

                if self.save_freq > 0:
                    if epoch_num % self.save_freq == 0:
                        self.logger.info(f"Epoch {epoch_num}")
                        self.save(
                            os.path.join(self.nn_dir, f"checkpoint_{epoch_num:06d}")
                        )
                        self.eval()

                if epoch_num > self.max_epochs:
                    self.save(
                        os.path.join(
                            self.nn_dir,
                            "last_"
                            + self.config["name"]
                            + "ep"
                            + str(epoch_num)
                            + "rew"
                            + str(mean_rewards),
                        )
                    )
                    self.logger.info("MAX EPOCHS NUM!")
                    should_exit = True

                update_time = 0
            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit).float()
                self.hvd.broadcast_value(should_exit_t, "should_exit")
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

    # def play_steps(self):
    #     epinfos = []
    #     update_list = self.update_list
    #
    #     step_time = 0.0
    #
    #     for n in range(self.horizon_length):
    #         self.obs, done_env_ids = self._env_reset_done()
    #         if self.use_action_masks:
    #             masks = self.vec_env.get_action_masks()
    #             res_dict = self.get_masked_action_values(self.obs, masks)
    #         else:
    #             res_dict = self.get_action_values(self.obs)
    #
    #         self.experience_buffer.update_data("obses", n, self.obs["obs"])
    #         self.experience_buffer.update_data("dones", n, self.dones)
    #
    #         for k in update_list:
    #             self.experience_buffer.update_data(k, n, res_dict[k])
    #         if self.has_central_value:
    #             self.experience_buffer.update_data("states", n, self.obs["states"])
    #
    #         step_time_start = time.time()
    #         self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
    #         step_time_end = time.time()
    #
    #         step_time += step_time_end - step_time_start
    #
    #         shaped_rewards = self.rewards_shaper(rewards)
    #
    #         if self.value_bootstrap and "time_outs" in infos:
    #             shaped_rewards += (
    #                 self.gamma
    #                 * res_dict["values"]
    #                 * self.cast_obs(infos["time_outs"]).unsqueeze(1).float()
    #             )
    #
    #         self.experience_buffer.update_data("rewards", n, shaped_rewards)
    #
    #         # terminated = infos['terminate'].float()
    #         # terminated = terminated.unsqueeze(-1)
    #         # next_vals = self._eval_critic(self.obs)
    #         # next_vals *= (1.0 - terminated)
    #         # self.experience_buffer.update_data('next_values', n, next_vals)
    #
    #         self.current_rewards += rewards
    #         self.current_lengths += 1
    #         all_done_indices = self.dones.nonzero(as_tuple=False)
    #         done_indices = all_done_indices[:: self.num_agents]
    #
    #         self.game_rewards.update(self.current_rewards[done_indices])
    #         self.game_lengths.update(self.current_lengths[done_indices])
    #         self.algo_observer.process_infos(infos, done_indices)
    #
    #         not_dones = 1.0 - self.dones.float()
    #
    #         self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
    #         self.current_lengths = self.current_lengths * not_dones
    #
    #     last_values = self.get_values(self.obs)
    #
    #     fdones = self.dones.float()
    #     mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
    #     mb_values = self.experience_buffer.tensor_dict["values"]
    #     mb_rewards = self.experience_buffer.tensor_dict["rewards"]
    #     mb_advs = self.discount_values(
    #         fdones, last_values, mb_fdones, mb_values, mb_rewards
    #     )
    #     mb_returns = mb_advs + mb_values
    #
    #     batch_dict = self.experience_buffer.get_transformed_list(
    #         swap_and_flatten01, self.tensor_list
    #     )
    #     batch_dict["returns"] = swap_and_flatten01(mb_returns)
    #     batch_dict["played_frames"] = self.batch_size
    #     batch_dict["step_time"] = step_time
    #
    #     return batch_dict

    def eval(self):
        self.logger.info("Hello eval")
        # max_steps = 200
        # imgs = []
        # red_pos = []
        # # red_rot = []
        # blue_pos = []
        # blue_rot = []
        # steps = 0
        # done_indices = torch.arange(
        #     self.num_actors, dtype=torch.int, device=self.device
        # )
        # posrot_output_dir = os.path.join(self.network_path, "posrot")
        # os.makedirs(os.path.join(posrot_output_dir), exist_ok=True)
        # obs, done_env_ids = self._env_reset_done()
        # with torch.no_grad():
        #     for _ in tqdm(range(max_steps)):
        #         res_dict = self.get_action_values(obs)
        #         obs, r, done, info = self.env_step(res_dict["mus"])
        #
        #         red_pos.append(self.vec_env.env.red_rb_xyz[0])
        #         blue_pos.append(self.vec_env.env.blue_rb_xyz[0])
        #         blue_rot.append(self.vec_env.env.blue_rb_rot[0])
        #
        #         # for i in range(24):
        #         #     red_actor = self.red_actors[i]
        #         #     blue_actor = self.blue_actors[i]
        #         #
        #         #     m = np.eye(4)
        #         #     m[:3, 3] = red_pos[-1][i]
        #         #     red_actor.user_matrix = m
        #         #
        #         #     m = np.eye(4)
        #         #     m[:3, 3] = blue_pos[-1][i]
        #         #     blue_actor.user_matrix = m
        #         #
        #         # self.pl.render()
        #         # img = np.array(self.pl.screenshot())
        #         # imgs.append(img)
        #
        #         steps += 1
        #         done *= 0
        #         all_done_indices = done.nonzero(as_tuple=False)
        #         done_indices = all_done_indices[:: self.num_agents]
        #         done_count = len(done_indices)
        #         done_indices = done_indices[:, 0]
        #
        # # imgs = np.stack(imgs, axis=0)[None]
        # # self.writer.add_video(
        # #     f"video", imgs, dataformats="NTHWC", fps=15, global_step=self.epoch_num
        # # )
        #
        # red_pos = torch.stack(red_pos, dim=0).detach().cpu().numpy()
        # blue_pos = torch.stack(blue_pos, dim=0).detach().cpu().numpy()
        # blue_rot = torch.stack(blue_rot, dim=0).detach().cpu().numpy()
        #
        # d = {
        #     "red_pos": red_pos,
        #     "blue_pos": blue_pos,
        #     "blue_rot": blue_rot,
        # }
        # out_dir = os.path.join(
        #     proj_dir, "out", self.config["run_name"], self.config["out_name"]
        # )
        # out_path = os.path.join(out_dir, f"posrot_{self.epoch_num:06d}.pkl")
        # torch.save(d, out_path)
        #
        # return {}
