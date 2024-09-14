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


class CopycatDataset(Dataset):
    def __init__(self, data_dir, train_idx):
        self.data_dir = data_dir
        self.data_filenames = sorted(glob.glob(f"{data_dir}/[0-9]*.pkl"))
        self.lengths = torch.load(f"{data_dir}/lengths.pkl")

        if train_idx is not None:
            self.data_filenames = [self.data_filenames[i] for i in train_idx]
            self.lengths = [self.lengths[i] for i in train_idx]

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        filename = self.data_filenames[idx]
        d = torch.load(filename)
        return d


def copycat_collate_fn(batch):
    n_sequences = len(batch)
    dd = {}
    for k in list(batch[0].keys())[1:-1]:
        dd[k] = []
        for d in batch:
            dd[k].append(d[k])

        # Apply nanpads
        all_lengths = torch.tensor([a.shape[0] for a in dd[k]])
        max_seq_len = all_lengths.max()
        lengths_to_go = max_seq_len - all_lengths

        nanpads = [
            torch.ones((lengths_to_go[i], *(dd[k][i].shape[1:]))) * torch.nan
            for i in range(len(batch))
        ]
        padded_tensors = [
            torch.cat([a, nanpads[i]], dim=0) for i, a in enumerate(dd[k])
        ]
        stacked_tensors = torch.stack(padded_tensors, dim=0)
        dd[k] = stacked_tensors
    dd["lengths"] = all_lengths
    return dd


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

    data_dir = f"{proj_dir}/data/nami/AMASS"
    copycat_path = "torchready_v5_dof_ar_walks.pkl"
    d = torch.load(copycat_path)
    # with open(copycat_path, "rb") as f:
    #     d = joblib.load(f)
    logger.info(f"Loaded copycat data from {copycat_path}")

    n_total_epochs = args.n_total_epochs
    pbar = tqdm(total=n_total_epochs)
    n_total_epochs = args.n_total_epochs
    epochs_elapsed = 0
    batch_size = args.batch_size
    model = None

    k = list(d.keys())[0]
    rb_vel_inv = [t.reshape(-1, 24, 3) for t in d["rb_vel_inv"]]
    rb_root_vel_inv = [t.reshape(-1, 24, 3)[..., 0, :] for t in rb_vel_inv]
    rb_ang_vel = [t.reshape(-1, 24, 3) for t in d["rb_ang"]]
    rb_root_ang_vel = [t[..., 0, :] for t in rb_ang_vel]

    
    xyz = rb_root_vel_inv[0][:, :2]
    quat = rb_root_ang_vel[0]
    #expm = quat_to_expm(quat)
    xyzexpms = np.concatenate([xyz, quat], axis=1)
    xyzexpms = torch.as_tensor(xyzexpms, dtype=torch.float, device=device)

    # For sanity check
    # xyzexpms = xyzexpms[170:]
    # tmp = xyzexpms[..., 2] * 1
    # xyzexpms[:] = 0
    # xyzexpms[..., 2] = tmp

    lengths = np.array([xyzexpms.shape[1]])
    segment_length = 10
    save_every = 300 #n_total_epochs // 5
    n_warmup_epochs = save_every // 10
    eval_every = save_every // 100
    lengths = torch.as_tensor(lengths, device=device)

    while epochs_elapsed <= n_total_epochs:
        if model is None:
            logger.info(f"Initializing a model")
            model = ConditionalVAEWithPrior(
                nail_size=10,
                hidden_size=args.hidden_size,
                latent_size=args.latent_size,
                cond_size=10,
            )
            model = model.to(device)
            model.nail_rms.update(xyzexpms.reshape(-1, xyzexpms.shape[-1]))
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
            t_start = (
                torch.rand((batch_size,), device=device) * (lengths - segment_length)
            ).to(dtype=torch.long)
            # t_start[:] = 150
            segment_ts = torch.arange(segment_length, device=device) + t_start[:, None]
            x_segment = torch.take_along_dim(xyzexpms, segment_ts[:, :, None], dim=1)
            x_hats = [
                # torch.ones_like(x_segment[:, [0]]) * torch.nan for _ in range(1)
                x_segment[:, [0]]
            ]  # To start with (t, t) when at the beginning of segment (can we do better?)
            rollout_xins = []
            rollout_wins = []
            rollout_xgts = []
            with torch.no_grad():
                for t in range(1, segment_length):
                    # Fetch and formulate VAE input
                    x_in = x_segment[:, [t]]
                    x_gt = x_segment[:, [t]]
                    w_in = torch.cat(x_hats[-1:], dim=1)

                    # Use VAE to generate
                    _, x_hat, _, _, _, _ = model.forward(x_in, w_in)
                    x_hats.append(x_hat)

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
        t_start = (
            torch.rand((batch_size,), device=device) * (lengths - segment_length)
        ).to(dtype=torch.long)
        # t_start[:] = 150
        segment_ts = torch.arange(segment_length, device=device) + t_start[:, None]
        x_segment = torch.take_along_dim(xyzexpms, segment_ts[:, :, None], dim=1)
        x_hats = [
            # torch.ones_like(x_segment[:, [0]]) * torch.nan for _ in range(1)
            x_segment[:, [0]]
        ]  # To start with (t, t) when at the beginning of segment (can we do better?)
        rollout_xins = []
        rollout_wins = []
        rollout_xgts = []
        with torch.no_grad():
            for t in range(1, segment_length):
                # Fetch and formulate VAE input
                x_in = x_segment[:, [t]]
                x_gt = x_segment[:, [t]]
                w_in = torch.cat(x_hats[-1:], dim=1)

                # Use VAE to generate
                _, x_hat, _, _, _, _ = model.forward(x_in, w_in)
                x_hats.append(x_hat)

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
