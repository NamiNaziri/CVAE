##NOTE: this is trained on multiple animations



import os
import sys

# sys.path.append(os.path.abspath("../.."))
current_dir = os.path.dirname(os.path.abspath(__file__))
target_folder_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Add the target folder to the system path
sys.path.insert(0, target_folder_path)


import pauli
from mlexp_utils import my_logging
from mlexp_utils.dirs import proj_dir
#import torch_nets
from torch_nets import PartWiseVAE, ConditionalVAE, ThroughDataset, ConditionalPartWiseVAE, ARConditionalPartWiseVAE
# import debugpy

# debugpy.listen(5679)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from argparse import ArgumentParser
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random

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
    writer.add_text("args", str(args))

    n_epochs = 50

    current_path = os.getcwd()
    print("Current Path:", current_path)

    if(not args.use_cache):
        #pkl_file_path = "torchready_v5_dof_ar_circles.pkl"#f"torchready_v5_dof_1k.pkl"
        #pkl_file_path = "torchready_v5_dof_ar_walks.pkl"#f"torchready_v5_dof_1k.pkl"
        pkl_file_path = "torchready_v5_dof_ar_dance_and_walk.pkl"#f"torchready_v5_dof_1k.pkl"
        
        data = torch.load(pkl_file_path)

        # pkl_file_path = f"torchready_v5_dof_ar_1k.pkl"
        # data1 = torch.load(pkl_file_path)

        # for key, value in data.items():
        #     data[key]= data[key] + data1[key]

            
        
        
        rb_rot_sixd_inv = data["rb_rot_sixd_inv"]
        rb_rot_sixd_inv = [t.reshape(-1, 24, 6) for t in rb_rot_sixd_inv]
        dof_pos = [t.reshape(-1, 23, 3) for t in data["dof_pos"]]

        rb_rot_sixd = [t.reshape(-1, 24, 6) for t in data["rb_rot_sixd"]]
        rb_pos_inv = [t.reshape(-1, 24, 3) for t in data["rb_pos_inv"]]
        rb_body_pos_inv = [t.reshape(-1, 24, 3)[..., 1:, :] for t in data["rb_pos_inv"]]
        rb_pos = [t.reshape(-1, 24, 3) for t in data["rb_pos"]]
        rb_root_pos = [t[:, 0] for t in rb_pos]
        rb_root_rot_sixd_inv = [t[:, 0] for t in rb_rot_sixd_inv]
        rb_vel = [t.reshape(-1, 24, 3) for t in data["rb_vel"]]
        rb_root_vel_xy = [t.reshape(-1, 24, 3)[..., 0, :2] for t in rb_vel]
        rb_vel_inv = [t.reshape(-1, 24, 3) for t in data["rb_vel_inv"]]
        rb_body_vel_inv = [t.reshape(-1, 24, 3)[..., 1:, :] for t in rb_vel_inv]
        rb_root_vel_inv = [t.reshape(-1, 24, 3)[..., 0, :] for t in rb_vel_inv]
        rb_ang_vel = [t.reshape(-1, 24, 3) for t in data["rb_ang"]]
        rb_root_ang_vel = [t[..., 0, [2]] for t in rb_ang_vel] #only the yaw angular velocity is neede


        lengths = torch.tensor([t.shape[1] for t in data["rb_rot_sixd_inv"]])
        max_seq_len = lengths.max()
        lengths_to_go = max_seq_len - lengths
        torch.save(lengths, "lengths.pkl")

    else:
        lengths = torch.load("lengths.pkl")
    
    # dof_pos = data["dof_pos"].reshape(-1, 23, 3)

    # rb_rot_sixd = data["rb_rot_sixd"]
    # rb_pos_inv = data["rb_pos_inv"].reshape(-1, 24, 3)
    # rb_pos = data["rb_pos"].reshape(-1, 24, 3)
    # rb_root_pos = rb_pos[:, 0]
    # rb_root_rot_sixd_inv = rb_rot_sixd_inv[:, 0]
    # rb_vel = data["rb_vel"].reshape(-1, 24, 3)
    # rb_root_vel_xy = data["rb_vel"].reshape(-1, 24, 3)[..., 0, :2]
    # rb_vel_inv = data["rb_vel_inv"].reshape(-1, 24, 3)
    # rb_root_vel_inv = data["rb_vel_inv"].reshape(-1, 24, 3)[..., 0, :]
    # rb_ang_vel = data["rb_ang"].reshape(-1, 24, 3)
    # rb_root_ang_vel = rb_ang_vel[..., 0, :]

    
    #
    body_names = ["L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
                       "Torso","Spine","Chest","Neck","Head","L_Thorax","L_Shoulder","L_Elbow",
                       "L_Wrist","L_Hand","R_Thorax","R_Shoulder","R_Elbow","R_Wrist","R_Hand",]


    upper_body = [ "Torso", "Spine","Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"]
    lower_body = [ "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe", ]

    left_arm = ["L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand"]
    right_arm=["R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"]
    left_leg=["L_Hip", "L_Knee", "L_Ankle", "L_Toe"]
    right_leg= ["R_Hip", "R_Knee", "R_Ankle", "R_Toe"]
    main_body=[ "Torso", "Spine", "Chest", "Neck", "Head"]



    chains=[lower_body, upper_body ]
    
    #NOTE this is a very important note! the order of these part are very important and it should be the same order as of the body names.
    #TODO implement a way to be able to have different types
    #chains=[left_leg, right_leg, main_body, left_arm, right_arm, ]
    chains_indecies = []

    for chain_idx, chain in enumerate(chains):
        chains_indecies.append([])
        for bodypart in chain:
            chains_indecies[chain_idx].append(body_names.index(bodypart))

    # xs = np.concatenate([rb_rot_sixd_inv[:, 1:], rb_ang_vel[:, 1:]], axis=-1)
    #TODO I am including the root here, but we can remove it in our body part (remove the Pelvis from the lower body list)
    #[rb_root_vel_xy, rb_root_ang_vel, rb_pos_inv, rb_vel_inv, rb_pos]
    
    #xs = np.concatenate([rb_pos_inv[:, 1:], rb_vel_inv[:, 1:],  dof_pos], axis=-1)
    #xs = np.concatenate([ dof_pos], axis=-1)
    #ys_lower = np.concatenate([xs_root, xs[:,chains_indecies[0]].reshape(xs.shape[0], -1 )], axis=-1)
    if(args.use_cache):
        xs = torch.load("xs_cached.pkl")
        ys = torch.load("ys_cached.pkl")
    else:
        
        #xs = [np.concatenate([t1[...,:2], t2], axis=-1) for (t1,t2) in zip(rb_root_vel_inv,rb_root_ang_vel)]
        xs = [np.concatenate([t2[...,:2], t3 ,t1.reshape(t1.shape[0],-1), t4.reshape(t4.shape[0],-1),t5.reshape(t5.shape[0],-1) ], axis=-1) for (t1,t2,t3,t4,t5) in zip(dof_pos,rb_root_vel_inv,rb_root_ang_vel,rb_body_vel_inv,rb_body_pos_inv  )]
        #self.rb_root_vel_inv[...,:2], self.rb_root_ang_vel,self.dof_pos.reshape(-1,69)
        ys = xs * 1 #[np.concatenate([t2[...,:2], t3 ,t1.reshape(t1.shape[0],-1)], axis=-1) for (t1,t2,t3) in zip(dof_pos,rb_root_vel_inv,rb_root_ang_vel)]
        #        ys = [np.concatenate([t2[...,:2], t3 ], axis=-1) for (t2,t3) in zip(rb_root_vel_inv,rb_root_ang_vel)] #root only

        xs = [t.reshape(t.shape[0], -1) for t in xs]
        xs = [torch.tensor(t, dtype=torch.float) for t in xs]

        ys = [t.reshape(t.shape[0], -1) for t in ys]
        ys = [torch.tensor(t, dtype=torch.float) for t in ys]


        xs_nanpads = [
            torch.ones((l, t.shape[-1]))* 0
            for (l,t) in zip(lengths_to_go, xs)
        ]
        ys_nanpads = [
            torch.ones((l, t.shape[-1]))* 0
            for (l,t) in zip(lengths_to_go, ys)
        ]

        xs_stack = torch.stack([torch.cat([t1,t2], dim = 0) for (t1,t2) in zip(xs, xs_nanpads)], dim =0)
        ys_stack = torch.stack([torch.cat([t1,t2], dim = 0) for (t1,t2) in zip(ys, ys_nanpads)], dim =0)
        
        xs = xs_stack * 1
        ys = ys_stack * 1



        torch.save(xs, "xs_cached.pkl")
        torch.save(ys, "ys_cached.pkl")

    
    #ys_lower = np.roll(ys_lower, shift=1, axis=0) # shift to the right. this way, it ys[t] contains the pose of t-1
    #NOTE when we want to remove the pelvix(root) from body parts, the (len(body_names)) should be changed to 23
    hidden_sizes = []
    latent_sizes = []
    # nail_sizes = []
    # for chain_idx, chain in enumerate(chains):
    #     percentage = len(chain) / len(body_names)
    #     hidden_sizes.append(int (percentage * args.hidden_size))
    #     latent_sizes.append(int (percentage * args.latent_size))
    #     nail_sizes.append(len(chain) * xs.shape[-1])

    # nail_sizes[0] += 5
    # cond_sizes = nail_sizes * 1
    # #cond_sizes[0] =  nail_sizes[0] # lower body: condition on prev lower body + root. done this in prev line
    # cond_sizes[1] = nail_sizes[0] # upper body: condition on the lower body
    
    # latent_sizes[0] = 8
    # latent_sizes[1] = 8
    print(latent_sizes)
    #NOTE no need to reshape, since I do this inside the PVAE

    steps_after_50_epochs = 0
    n_train = int(xs.shape[0] * 0.99)
    #n_valid = xs.shape[0] - n_train

    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    ###train_idxs = torch.tensor(np.random.choice(xs.shape[0], n_train, replace=False))
    train_idxs = torch.tensor(np.arange(xs.shape[0]-1))
    #train_idxs = train_idxs[[0]]
    # train_idxs = np.arange(10)  # use this for first few frame training
    ###valid_idxs = torch.tensor(np.setdiff1d(np.arange(xs.shape[0]), train_idxs))
    valid_idxs = torch.tensor([xs.shape[0] - 1])
    xs_train = xs.cuda() #TODO use train_idx when we have validation. the first index should be the animation idx not the frame idx!
    #xs_train = [t.cuda() for t in xs_train]
    ys_train = ys.cuda()
    #ys_train = [t.cuda() for t in ys_train]
    #orig_xs_train = xs[train_idxs] * 1.0

    # print(valid_idxs.shape)
    # xs_valid = (
    #     xs[valid_idxs[: int(valid_idxs.shape[0] / 4)]] * 1.0
    # )  # NOTE is this correct
    # xs_valid = torch.as_tensor(xs[valid_idxs], dtype=torch.float)
    # ys_valid = torch.as_tensor(ys[valid_idxs], dtype=torch.float)

    # print(torch.max(xs_train, dim = -1))
    # xs_train = xs_train.reshape(xs_train.shape[0], 23,  -1)
    # xs_valid = xs_valid.reshape(xs_valid.shape[0], 23, -1)
    # init_lr = 3e-4
    # init_lr = 5e-5 #smaller init because of nans
    #init_lr = 5e-5  # smaller init because of nans
    init_lr = 1e-4
    init_kd_weight = 0
    kl_weight =  init_kd_weight
    # init_lr = 5e-7  # smaller init because of nans


    if args.checkpoint_path is None:
        model = ConditionalVAE(
            xs[0].shape[-1],
            args.hidden_size,
            args.latent_size,
            ys[0].shape[-1] * args.num_history, # condition size
            ys[0].shape[-1], # out size
        )
        model = model.cuda()
        #model.nail_rms = model.nail_rms.cuda()
        optimizer = Adam(model.parameters(), init_lr)
        global_steps_elapsed = 0
        epochs_elapsed = 0
    else:
        print("loading from checkpoint.")
        print(args.checkpoint_path)
        model_dict = torch.load(args.checkpoint_path)
        model = pauli.load(model_dict)
        print(type(model))
        model.load_state_dict(model_dict["model_state_dict"])
        model = model.cuda()
        #model.eval()
        #model.nail_rms = model.nail_rms.cuda()
        optimizer = Adam(model.parameters(), init_lr)
        optimizer.load_state_dict(model_dict["optimizer_state_dict"])
        global_steps_elapsed = model_dict["global_steps_elapsed"]
        epochs_elapsed = model_dict["epochs_elapsed"]

    save_every = 20
    eval_every = 10

    epoch_num = 0
    

    batch_train_idxs = torch.arange(len(train_idxs)).repeat_interleave(lengths[train_idxs.tolist()]- 10 - 2 ).to(dtype=torch.long)
    train_lengths = lengths[train_idxs.tolist()]
    t_train_start = torch.cat([torch.arange(length- 10 - 2) + 1 for length in train_lengths]).to(dtype=torch.long)
    full_epoch = torch.stack([train_idxs[batch_train_idxs], t_train_start], dim=-1)


    # save_every = 10
    # n_total_steps = int(1e7)
    n_total_steps = int(2.5e15)
    #pbar = tqdm(total=n_total_steps)
    while global_steps_elapsed <= n_total_steps:

        if epochs_elapsed % eval_every == 0 or global_steps_elapsed >= n_total_steps:
            if epochs_elapsed % save_every == 0 or global_steps_elapsed >= n_total_steps:
                d = {
                    **pauli.dump(model),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_steps_elapsed": global_steps_elapsed,
                    "epochs_elapsed": epochs_elapsed,
                    "args": args,
                }
                save_path = f"{out_dir}/vae_{epochs_elapsed:06d}.pkl"
                torch.save(d, save_path)
                logger.info(f"Saved to {save_path}")

            #batch_valid_idxs = (torch.rand(args.sample_batch_size) * len(valid_idxs)).to(dtype=torch.long)
            steps = min(100, int(lengths[valid_idxs.tolist()].item() / 2))
            batch_valid_idxs = torch.arange(len(valid_idxs)).repeat_interleave(lengths[valid_idxs.tolist()]- steps - 2 ).to(dtype=torch.long)
                            
            valid_lengths = lengths[valid_idxs.tolist()] #TODO: this is temp for single animation with length of 800
            # t_start = (
            #     torch.rand((args.sample_batch_size,))
            #     * (valid_lengths[batch_valid_idxs] - 10 - 2) # making sure it won't go out of animation length
            #     + 1
            # ).to(dtype=torch.long)

            t_start = torch.cat([torch.arange(length- steps - 2) + 1 for length in valid_lengths]).to(dtype=torch.long)

            it = torch.stack([valid_idxs[batch_valid_idxs], t_start], dim=-1)

            y_hat_start = ys_train[it[:, 0], it[:, 1]]

            y_hats = [
                y_hat_start
                for _ in range(args.num_history) #example: if num_history = 3, it means that it is conditioned on 3 previous frames
            ]

            rollout_xins = []
            rollout_youts = []
            rollout_gts = []
            with torch.no_grad():
                for t in range(steps):
                    x_in = xs_train[it[:, 0], it[:, 1] + t]
                    w_in = torch.stack(y_hats[-args.num_history :], 1)
                    w_in = w_in.reshape(w_in.shape[0], -1)
                    y_tar = ys_train[it[:, 0], it[:, 1] + t]
                    z, decoded, mu, log_var = model(x_in, w_in, False)
                    y_hats.append(decoded)

                    rollout_xins.append(x_in)
                    rollout_youts.append(decoded)
                    rollout_gts.append(y_tar)

                rollout_xins = torch.concatenate(rollout_xins, dim=0)
                rollout_youts = torch.concatenate(rollout_youts, dim=0)
                rollout_gts = torch.concatenate(rollout_gts, dim=0)
                
                #KLD only for last one
                Eval_KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                mse_loss = torch.nn.MSELoss()
                Eval_MSE = mse_loss(rollout_youts, rollout_gts)
                writer.add_scalar("valid/MSEloss", Eval_MSE.item(), global_steps_elapsed)
                writer.add_scalar("valid/KLDloss", Eval_KLD.item(), global_steps_elapsed)
                writer.add_scalar("valid/mu", z.mean().item(), global_steps_elapsed)
                writer.add_scalar("valid/sigma", z.std().item(), global_steps_elapsed)
                logger.info(f"Epoch {epochs_elapsed} MSELoss: {Eval_MSE.item()} mu: {z.mean().item()} sigma: {z.std().item()}")
                writer.add_scalar(
                    "valid/MSEloss/epochs", Eval_MSE.item(), epochs_elapsed
                )

        #if ((epochs_elapsed > 100) and (epochs_elapsed % 205 == 0)):
        # if ((epochs_elapsed > 100) and (epochs_elapsed % 4005 == 0)):
        #     args.scheduled_sampling_length = min(args.scheduled_sampling_length + 1 , 10)

        

        # lr = init_lr
        end_lr = 1e-7
        n_anneal_steps = 5000000
        ratio = min(1, global_steps_elapsed / n_anneal_steps)
        lr = init_lr * (1 - ratio) + end_lr * ratio

        if(epochs_elapsed == 50):
            steps_after_50_epochs = global_steps_elapsed
        if(epochs_elapsed > 50):
            kl_ratio = min(1, (global_steps_elapsed - steps_after_50_epochs) / n_anneal_steps)
            kl_weight =  init_kd_weight  * (1 - kl_ratio) + args.kld_weight * kl_ratio
        # lr = init_lr * (1 - (min(1, global_steps_elapsed / n_total_steps)))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        #TODO xs_train has additional zeros! 
        # if epochs_elapsed == 0:
        #     model.nail_rms.update(xs_train.reshape(-1, xs_train.shape[-1]))
        #     model.cond_rms.update(xs_train.reshape(-1, xs_train.shape[-1]))
        #     logger.info(f"Normalized input")

        if global_steps_elapsed < n_total_steps:


            sampler = BatchSampler(
                SubsetRandomSampler(full_epoch),
                args.sample_batch_size,
                drop_last=False,
            )

            num_mini_batch = 1
            for num_mini_batch, indices in enumerate(tqdm(sampler, desc="Epoch Progress")):

                it = torch.stack(indices)

                y_hat_start = ys_train[it[:, 0], it[:, 1]]

                y_hats = [
                    y_hat_start
                    for _ in range(args.num_history) #example: if num_history = 3, it means that it is conditioned on 3 previous frames
                ]
                
                for t in range(args.scheduled_sampling_length):
                    x_in = xs_train[it[:, 0], it[:, 1] + t] * 1
                    w_in = torch.stack(y_hats[-args.num_history :], 1) * 1
                    w_in = w_in.reshape(w_in.shape[0], -1) * 1
                    y_tar = ys_train[it[:, 0], it[:, 1] + t] * 1

                    teacher_forcing = 25
                    if(epochs_elapsed < teacher_forcing or (epochs_elapsed < 50 and (random.uniform(0, 1) >=  (epochs_elapsed - teacher_forcing) / 25))):
                        w_in = ys_train[it[:, 0], it[:, 1] + t - 1]

                    z, decoded, mu, log_var = model(x_in, w_in, False)  # don't collect logits here
                    # y_tar = train_y[it[:, 0], it[:, 1] + t + 1][:, None]
                    
                    y_hats.append(decoded.detach())

                    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                    MSE = torch.nn.functional.mse_loss(decoded, y_tar)
                    loss = MSE + kl_weight * KLD

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    global_steps_elapsed += 1

                    if global_steps_elapsed >= n_total_steps:
                        break
                writer.add_scalar("train/loss", loss.item(), global_steps_elapsed)
                writer.add_scalar("train/MSE", (MSE).item(), global_steps_elapsed)
                writer.add_scalar("train/KLD", (KLD).item(), global_steps_elapsed)
                writer.add_scalar("train/lr", lr, global_steps_elapsed)
                writer.add_scalar("train/kld_weight", kl_weight, global_steps_elapsed)
            epoch_num += 1
            epochs_elapsed += 1
            writer.add_scalar("train/MSE/epochs", (MSE).item(), epochs_elapsed)
            writer.add_scalar("train/schedule_sampling/epochs", args.scheduled_sampling_length, epochs_elapsed)
            writer.add_scalar(
                    "train/loss/epochs", loss.item(), epochs_elapsed
                )
            logger.info(
                f"Epoch {epochs_elapsed}:\tMSE: {MSE.item():.3f}\ kld loss: {KLD.item():.3f} \t SS: {args.scheduled_sampling_length:.3f} \t mu: {z.mean().item():.2e} sigma: {z.std().item():.2e}"
            )
        else:
            break

    #pbar.close()
    writer.flush()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--kld_weight", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=800)
    parser.add_argument("--sample_batch_size", type=int, default=800)
    parser.add_argument("--scheduled_sampling_length", type=int, default=10)
    parser.add_argument("--num_history", type=int, default=1)

    
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--override_init_yes", action="store_true")
    parser.add_argument("--debug_yes", action="store_true")
    parser.add_argument("--use_cache", action="store_true",  default=False)
    args = parser.parse_args()

    main()
