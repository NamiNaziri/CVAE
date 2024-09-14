import os
import sys

import pauli
from mlexp_utils import my_logging
from mlexp_utils.dirs import proj_dir
from torch_nets import ConditionalVAE, ThroughDataset

sys.path.append(os.path.abspath("./src"))

from argparse import ArgumentParser
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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

    pkl_file_path = f"torchready_v5_locomo.pkl"
    data = torch.load(pkl_file_path)

    rb_rot_sixd_inv = data["rb_rot_sixd_inv"]
    rb_rot_sixd_inv = rb_rot_sixd_inv.reshape(-1, 24, 6)
    rb_rot_sixd = data["rb_rot_sixd"]
    rb_pos_inv = data["rb_pos_inv"].reshape(-1, 24, 3)
    rb_pos = data["rb_pos"].reshape(-1, 24, 3)
    rb_root_pos = rb_pos[:, 0]
    rb_root_rot_sixd_inv = rb_rot_sixd_inv[:, 0]
    rb_vel_inv = data["rb_vel_inv"].reshape(-1, 24, 3)
    rb_root_vel_inv = data["rb_vel_inv"].reshape(-1, 24, 3)[..., 0, :]
    rb_ang_vel = data["rb_ang"].reshape(-1, 24, 3)
    rb_root_ang_vel = rb_ang_vel[..., 0, :]

    # xs = np.concatenate([rb_rot_sixd_inv[:, 1:], rb_ang_vel[:, 1:]], axis=-1)
    xs = np.concatenate([rb_rot_sixd_inv[:, 1:]], axis=-1)
    xs = xs.reshape(xs.shape[0], -1)
    # xs = torch.tensor(xs, dtype=torch.float)

    # xs_gt = np.concatenate([rb_pos_inv, rb_vel_inv], axis=-1)
    xs_gt = np.concatenate([rb_pos_inv], axis=-1)
    xs_gt = xs_gt.reshape(xs_gt.shape[0], -1)

    ys = np.concatenate(
        [
            rb_root_pos[..., [-1]],  # z position
            rb_root_rot_sixd_inv,
            # rb_root_vel_inv,
            # rb_root_ang_vel,
        ],
        axis=-1,
    )

    n_train = int(xs.shape[0] * 0.99)
    n_valid = xs.shape[0] - n_train

    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    train_idxs = np.random.choice(xs.shape[0], n_train, replace=False)
    # train_idxs = np.arange(10)  # use this for first few frame training
    valid_idxs = np.setdiff1d(np.arange(xs.shape[0]), train_idxs)
    xs_train = xs[train_idxs] * 1.0
    xs_gt_train = xs_gt[train_idxs] * 1.0
    ys_train = ys[train_idxs] * 1.0
    # print(valid_idxs.shape)
    # xs_valid = (
    #     xs[valid_idxs[: int(valid_idxs.shape[0] / 4)]] * 1.0
    # )  # NOTE is this correct
    xs_valid = torch.as_tensor(xs[valid_idxs], dtype=torch.float)
    ys_valid = torch.as_tensor(ys[valid_idxs], dtype=torch.float)
    xs_gt_valid = torch.as_tensor(xs_gt[valid_idxs], dtype=torch.float)

    # print(torch.max(xs_train, dim = -1))
    xs_train = xs_train.reshape(xs_train.shape[0], -1)
    xs_valid = xs_valid.reshape(xs_valid.shape[0], -1)
    # init_lr = 3e-4
    # init_lr = 5e-5 #smaller init because of nans
    init_lr = 5e-5  # smaller init because of nans
    # init_lr = 5e-7  # smaller init because of nans

    if args.checkpoint_path is None:
        model = ConditionalVAE(
            xs_train.shape[-1],
            args.hidden_size,
            args.latent_size,
            ys.shape[-1],
            xs_gt_train.shape[-1],
        )
        model = model.cuda()
        model.nail_rms = model.nail_rms.cuda()
        model.cond_rms = model.cond_rms.cuda()
        optimizer = Adam(model.parameters(), init_lr)
        global_steps_elapsed = 0
        epochs_elapsed = 0
    else:
        print("loading from checkpoint.")
        model_dict = torch.load(args.checkpoint_path)
        model = pauli.load(model_dict)
        model.load_state_dict(model_dict["model_state_dict"])
        model = model.cuda()
        model.nail_rms = model.nail_rms.cuda()
        model.cond_rms = model.cond_rms.cuda()
        optimizer = Adam(model.parameters(), init_lr)
        optimizer.load_state_dict(model_dict["optimizer_state_dict"])
        global_steps_elapsed = model_dict["global_steps_elapsed"]
        epochs_elapsed = model_dict["epochs_elapsed"]

    save_every = 5
    # save_every = 10
    # n_total_steps = int(1e7)
    n_total_steps = int(2.5e6)
    pbar = tqdm(total=n_total_steps)
    while global_steps_elapsed <= n_total_steps:

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

        with torch.no_grad():
            xs_valid = xs_valid.cuda()
            xs_gt_valid = xs_gt_valid.cuda()
            ys_valid = ys_valid.cuda()
            z, decoded, mu, log_var = model.forward(xs_valid, ys_valid, False)
            KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            MSE = torch.nn.functional.mse_loss(xs_gt_valid, decoded)
            loss = MSE + KLD * args.kld_weight

            writer.add_scalar("valid/MSEloss", MSE.item(), global_steps_elapsed)
            writer.add_scalar("valid/KLDloss", KLD.item(), global_steps_elapsed)

        logger.info(
            f"Epoch {epochs_elapsed}:\tMSE: {MSE.item():.3f}\tKLD: {KLD.item():.3f}"
        )

        # lr = init_lr
        end_lr = 1e-7
        n_anneal_steps = 250000
        ratio = min(1, global_steps_elapsed / n_anneal_steps)
        lr = init_lr * (1 - ratio) + end_lr * ratio
        # lr = init_lr * (1 - (min(1, global_steps_elapsed / n_total_steps)))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if global_steps_elapsed < n_total_steps:
            dataset = ThroughDataset(
                xs_train,
                ys_train,
                xs_gt_train,
            )
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            # dataloader = DataLoader(dataset, batch_size=4096, shuffle=False)
            # pbar = tqdm(total=len(dataset))

            for i, (x, y, x_gt) in enumerate(dataloader):
                x = x.pin_memory().to("cuda", non_blocking=True, dtype=torch.float)
                y = y.pin_memory().to("cuda", non_blocking=True, dtype=torch.float)
                x_gt = x_gt.pin_memory().to(
                    "cuda", non_blocking=True, dtype=torch.float
                )

                # torch.save(x, "x.pkl")
                # torch.save(y, "y.pkl")
                # torch.save(x_gt, "x_gt.pkl")

                # z, decoded, mu, log_var = model(x, y, True)
                z, decoded, mu, log_var = model(x, y, False)

                # Compute the loss and perform backpropagation
                # KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

                loss = (
                    torch.nn.functional.mse_loss(x_gt, decoded) + args.kld_weight * KLD
                )
                # loss += 1e-2 * torch.mean(encoded ** 2)
                # loss_function(decoded, x, mu, log_var, xs_train.shape[-1])#
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_steps_elapsed += 1

                pbar.update(1)
                pbar.set_postfix({"loss": loss.item(), "steps": global_steps_elapsed})
                if global_steps_elapsed >= n_total_steps:
                    break

            epochs_elapsed += 1
            writer.add_scalar("train/loss", loss.item(), global_steps_elapsed)
            writer.add_scalar("train/lr", lr, global_steps_elapsed)
        else:
            break

    pbar.close()
    writer.flush()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--kld_weight", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--override_init_yes", action="store_true")
    parser.add_argument("--debug_yes", action="store_true")
    args = parser.parse_args()

    main()
