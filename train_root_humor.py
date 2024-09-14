import glob
import os, sys
from argparse import ArgumentParser

current_dir = os.path.dirname(os.path.abspath(__file__))
target_folder_path = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Add the target folder to the system path
sys.path.insert(0, target_folder_path)


import joblib
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim import RAdam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import pauli
from mlexp_utils import my_logging
from mlexp_utils.dirs import proj_dir
from NH_torch_nets import ConditionalVAE, ConditionalWAE,ThroughDataset,  ConditionalVAEWithPrior

# import debugpy

# debugpy.listen(5679)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

body_names = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
]

upper_body = [
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
]
lower_body = [
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
]

left_arm = ["L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand"]
right_arm = ["R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"]
left_leg = ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe"]
right_leg = ["R_Hip", "R_Knee", "R_Ankle", "R_Toe"]
main_body = ["Torso", "Spine", "Chest", "Neck", "Head"]
upper_body_idxs = [body_names.index(b) for b in upper_body]
lower_body_idxs = [body_names.index(b) for b in lower_body]


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
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    device = torch.device("cuda")

    outdir = f"{proj_dir}/out/{args.run_name}/{args.out_name}"
    logdir = f"{proj_dir}/logdir/{args.run_name}/{args.out_name}"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    logger = my_logging.get_logger(f"{args.out_name}", logdir)
    logger.info(f"Starting")
    writer = SummaryWriter(logdir)
    writer.add_text("args", str(args))

    # data_dir = f"{proj_dir}/data/nami/AMASS"
    # copycat_path = os.path.join(data_dir, f"amass_copycat_take5_getup.pkl")
    # with open(copycat_path, "rb") as f:
    #     d = joblib.load(f)
    # logger.info(f"Loaded copycat data from {copycat_path}")

    n_total_epochs = args.n_total_epochs
    pbar = tqdm(total=n_total_epochs)
    n_total_epochs = args.n_total_epochs
    epochs_elapsed = 0
    batch_size = args.batch_size
    model = None

    pkl_file_path = "torchready_v5_dof_ar_walks.pkl"
    data = torch.load(pkl_file_path)
    rb_rot_sixd_inv = data["rb_rot_sixd_inv"]
    rb_rot_sixd_inv = [t.reshape(-1, 24, 6) for t in rb_rot_sixd_inv]
    dof_pos = [t.reshape(-1, 23, 3) for t in data["dof_pos"]]

    rb_rot_sixd = [t.reshape(-1, 24, 6) for t in data["rb_rot_sixd"]]
    rb_pos_inv = [t.reshape(-1, 24, 3) for t in data["rb_pos_inv"]]
    rb_pos = [t.reshape(-1, 24, 3) for t in data["rb_pos"]]
    rb_root_pos = [t[:, 0] for t in rb_pos]
    rb_root_rot_sixd_inv = [t[:, 0] for t in rb_rot_sixd_inv]
    rb_vel = [t.reshape(-1, 24, 3) for t in data["rb_vel"]]
    rb_root_vel_xy = [t.reshape(-1, 24, 3)[..., 0, :2] for t in rb_vel]
    rb_vel_inv = [t.reshape(-1, 24, 3) for t in data["rb_vel_inv"]]
    rb_root_vel_inv = [t.reshape(-1, 24, 3)[..., 0, :] for t in rb_vel_inv]
    rb_ang_vel = [t.reshape(-1, 24, 3) for t in data["rb_ang"]]
    rb_root_ang_vel = [t[..., 0, :] for t in rb_ang_vel]

    lengths = torch.tensor([t.shape[1] for t in data["rb_rot_sixd_inv"]])
    max_seq_len = lengths.max()
    lengths_to_go = max_seq_len - lengths

    # k = list(d.keys())[0]
    # xyz = d[k]["trans_orig"]
    # quat = d[k]["pose_quat_global"][:, 0]
    # expm = quat_to_expm(quat)
    # xyzexpms = np.concatenate([xyz, expm], axis=1)
    # xyzexpms = torch.as_tensor(xyzexpms, dtype=torch.float, device=device)

    # vs = xyzexpms * 0
    # vs[1:] = xyzexpms[1:] - xyzexpms[:-1]

    # # For 1-sequence version
    # xyzexpms = xyzexpms[None]
    # vs = vs[None]
    #xyzexpmvs = torch.cat([xyzexpms, vs], dim=-1)
    xs = [np.concatenate([t2[...,:2], t3], axis=-1) for (t1,t2,t3) in zip(dof_pos,rb_root_vel_inv,rb_root_ang_vel)]
    xs = [t.reshape(t.shape[0], -1) for t in xs]
    xs = [torch.tensor(t, dtype=torch.float) for t in xs]
    xs_nanpads = [
            torch.ones((l, t.shape[-1]))* 0
            for (l,t) in zip(lengths_to_go, xs)
        ]
    xs_stack = torch.stack([torch.cat([t1,t2], dim = 0) for (t1,t2) in zip(xs, xs_nanpads)], dim =0)
    xs = xs_stack * 1
    train_idxs = torch.tensor(np.arange(xs.shape[0]-1))
    valid_idxs = torch.tensor([xs.shape[0] - 1])



    xyzexpmvs = xs_stack.cuda() * 1
    #lengths = np.array([vs.shape[1]])
    segment_length = 10
    save_every = 300
    n_warmup_epochs = save_every // 10
    eval_every = save_every // 100
    #lengths = torch.as_tensor(lengths)

    while epochs_elapsed <= n_total_epochs:
        if model is None:
            logger.info(f"Initializing a model")
            model = ConditionalVAEWithPrior(
                nail_size=xyzexpmvs.shape[-1],
                hidden_size=args.hidden_size,
                latent_size=args.latent_size,
                cond_size=xyzexpmvs.shape[-1],
            )
            model = model.to(device)
            model.nail_rms.update(xyzexpmvs.reshape(-1, xyzexpmvs.shape[-1]))
            optimizer = RAdam(model.parameters(), lr=args.peak_lr)

        if epochs_elapsed % save_every == 0 or epochs_elapsed >= n_total_epochs:
            pauli_root = os.path.abspath(
                os.path.join(os.getcwd(), "src")
            )  # NOTE: assume that all scripts are run from the parent directory of src.
            model_d = {
                **pauli.dump(model, pauli_root),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epochs_elapsed": epochs_elapsed,
                "args": args,
            }
            save_path = f"{outdir}/wae_{epochs_elapsed:06d}.pkl"
            torch.save(model_d, save_path)
            logger.info(f"Saved to {save_path}")

        if epochs_elapsed >= n_total_epochs:
            break

        # Evaluation
        if epochs_elapsed % eval_every == 0:
            model.eval()
            
            # t_start[:] = 150
            steps = segment_length
            batch_valid_idxs = torch.arange(len(valid_idxs)).repeat_interleave(lengths[valid_idxs.tolist()]- steps - 2 ).to(dtype=torch.long)
            valid_lengths = lengths[valid_idxs.tolist()]
            t_start = torch.cat([torch.arange(length- steps - 2) + 1 for length in valid_lengths]).to(dtype=torch.long)
            it = torch.stack([valid_idxs[batch_valid_idxs], t_start], dim=-1)
            x_hat_start = xyzexpmvs[it[:, 0], it[:, 1]]
            x_hats = [
                x_hat_start
                for _ in range(1) #example: if num_history = 3, it means that it is conditioned on 3 previous frames
            ]


            rollout_xins = []
            rollout_wins = []
            rollout_xgts = []
            with torch.no_grad():
                for t in range(steps):
                    # Fetch and formulate VAE input
                    x_in = xyzexpmvs[it[:, 0], it[:, 1] + t]
                    x_gt = xyzexpmvs[it[:, 0], it[:, 1] + t]
                    w_in = torch.cat(x_hats[-1:], dim=1)

                    # Use VAE to generate
                    _, x_hat, _, _, _, _ = model.forward(x_in, w_in)
                    x_hats.append(x_hat[:,0])

                    rollout_xins.append(x_in)
                    rollout_wins.append(w_in)
                    rollout_xgts.append(x_gt)

                rollout_xins = torch.cat(rollout_xins, dim=0)
                rollout_wins = torch.cat(rollout_wins, dim=0)
                rollout_xgts = torch.cat(rollout_xgts, dim=0)

                (
                    rollout_zs,
                    rollout_xhats,
                    rollout_mus,
                    rollout_log_vars,
                    rollout_prior_mus,
                    rollout_prior_log_vars,
                ) = model.forward(rollout_xins, rollout_wins)
                recon_loss = torch.nn.functional.smooth_l1_loss(
                    rollout_xhats, rollout_xgts, reduction="mean"
                )
                my_dist = torch.distributions.Normal(
                    rollout_mus, rollout_log_vars.exp().sqrt()
                )
                prior_dist = torch.distributions.Normal(
                    rollout_prior_mus, rollout_prior_log_vars.exp().sqrt()
                )
                kld_loss = torch.distributions.kl.kl_divergence(
                    my_dist, prior_dist
                ).mean()

                loss = recon_loss + args.kld_weight * kld_loss

                writer.add_scalar("train/loss", loss.item(), epochs_elapsed)
                writer.add_scalar("train/ReconLoss", recon_loss.item(), epochs_elapsed)
                writer.add_scalar("train/KLDLoss", kld_loss.item(), epochs_elapsed)
                ratio_between_losses = np.maximum(
                    recon_loss.item() / kld_loss.item(),
                    kld_loss.item() / recon_loss.item(),
                )
                writer.add_scalar(
                    "train/ReconKLDRatio", ratio_between_losses, epochs_elapsed
                )
                logger.info(
                    f"Epoch {epochs_elapsed}: Loss: {loss.item():.2e} ReconLoss: {recon_loss.item():.2e} KLDLoss: {kld_loss.item():.2e} Ratio: {ratio_between_losses:.2e}"
                )

        model.train()
        # First, autoregressive collection of input
        # Segment formation based on randomly sampled t
        batch_train_idxs = (torch.rand(batch_size) * len(train_idxs)).to(dtype=torch.long)
        train_lengths = lengths[train_idxs.tolist()]
        t_start = (
                torch.rand((batch_size,))
                * (train_lengths[batch_train_idxs] - segment_length - 2) # making sure it won't go out of animation length
                + 1
            ).to(dtype=torch.long)
        it = torch.stack([train_idxs[batch_train_idxs], t_start], dim=-1)
        x_hat_start = xyzexpmvs[it[:, 0], it[:, 1]]

        x_hats = [
            x_hat_start
            for _ in range(1) #example: if num_history = 3, it means that it is conditioned on 3 previous frames
        ]

        rollout_xins = []
        rollout_wins = []
        rollout_xgts = []
        with torch.no_grad():
            for t in range(1, segment_length):
                # Fetch and formulate VAE input
                x_in = xyzexpmvs[it[:, 0], it[:, 1] + t]
                x_gt = xyzexpmvs[it[:, 0], it[:, 1] + t]
                w_in = torch.cat(x_hats[-1:], dim=1)

                # Use VAE to generate
                _, x_hat, _, _, _, _ = model.forward(x_in, w_in)
                x_hats.append(x_hat[:,0])

                rollout_xins.append(x_in)
                rollout_wins.append(w_in)
                rollout_xgts.append(x_gt)

        rollout_xins = torch.cat(rollout_xins, dim=0)
        rollout_wins = torch.cat(rollout_wins, dim=0)
        rollout_xgts = torch.cat(rollout_xgts, dim=0)

        # Then supervised learning from input
        dataset = ThroughDataset(rollout_xins, rollout_wins, rollout_xgts)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        min_lr = 0

        if epochs_elapsed % save_every < n_warmup_epochs:
            lr = min_lr + (args.peak_lr - min_lr) * np.clip(
                (epochs_elapsed % save_every) / n_warmup_epochs, 0, 1
            )
        else:
            # cosine decay
            lr = (
                min_lr
                + (args.peak_lr - min_lr)
                * (
                    1
                    + np.cos(
                        np.pi
                        * (epochs_elapsed % save_every - n_warmup_epochs)
                        / (save_every - n_warmup_epochs)
                    )
                )
                / 2
            )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        writer.add_scalar("train/lr", lr, epochs_elapsed)

        for x_in, w_in, x_gt in dataloader:
            x_in = x_in.to(device)
            w_in = w_in.to(device)
            x_gt = x_gt.to(device)

            z, x_hat, mu, log_var, prior_mu, prior_log_var = model.forward(x_in, w_in)

            my_dist = torch.distributions.Normal(mu, log_var.exp().sqrt())
            prior_dist = torch.distributions.Normal(
                prior_mu, prior_log_var.exp().sqrt()
            )

            recon_loss = torch.nn.functional.smooth_l1_loss(x_hat, x_gt)
            kld_loss = torch.distributions.kl.kl_divergence(my_dist, prior_dist).mean()
            loss = recon_loss + args.kld_weight * kld_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        epochs_elapsed += 1
        pbar.update(1)
        pbar.set_postfix({"loss": loss.item(), "epochs": epochs_elapsed})

    pbar.close()

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--kld_weight", type=float, default=1.0)
    parser.add_argument("--n_total_epochs", type=int, default=int(3e4))
    parser.add_argument("--continue_yes", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--peak_lr", type=float, default=3e-4)
    parser.add_argument("--debug_yes", action="store_true")
    parser.add_argument("--latent_size", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--fooling_loss_weight", type=float, default=1.0)
    args = parser.parse_args()

    main()
