import os
import sys

from mlexp_utils import my_logging
from mlexp_utils.dirs import proj_dir

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


class Sourceable:

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def get_src(self, depth, included: set = None):
        my_src = f"global {self.__class__.__name__}\n" + inspect.getsource(
            self.__class__
        )
        srcs = []
        if depth == 0:
            srcs.append("global Sourceable\n" + inspect.getsource(Sourceable))
            included = set()
        for child_module in self._modules:
            child_module_inst = getattr(self, child_module)
            if (
                isinstance(child_module_inst, Sourceable)
                and child_module_inst.__class__ not in included
            ):
                included.add(child_module_inst.__class__)
                srcs.append(child_module_inst.get_src(depth + 1, included))
        srcs.append(my_src)
        return "\n".join(srcs)


# Function to extract import statements from a given AST node
def extract_imports(node):
    imports = []
    for item in node.body:
        if isinstance(item, ast.Import):
            for alias in item.names:
                imports.append(f"import {alias.name}")
        elif isinstance(item, ast.ImportFrom):
            module_name = item.module
            for alias in item.names:
                imports.append(f"from {module_name} import {alias.name}")
    return imports


# Function to get import statements as a formatted string from a given file
def get_imports_as_string(file_path):
    with open(file_path, "r") as file:
        source_code = file.read()

    tree = ast.parse(source_code)
    imports = extract_imports(tree)

    imports_string = "\n".join(imports)
    return imports_string


class RunningMeanStd(nn.Module, Sourceable):
    def __init__(self, epsilon: float = 1e-4, shape=(), *args, **kwargs):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        super().__init__(*args, **kwargs)
        self.mean = nn.Parameter(
            torch.zeros(shape, dtype=torch.float), requires_grad=False
        )
        self.var = nn.Parameter(
            torch.ones(shape, dtype=torch.float), requires_grad=False
        )
        self.count = epsilon
        self.epsilon = nn.Parameter(torch.tensor(epsilon), requires_grad=False)

    def update(self, arr: torch.Tensor) -> None:
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean.data = new_mean
        self.var.data = new_var
        self.count = new_count

    def normalize(self, arr: torch.Tensor) -> torch.Tensor:
        return torch.clip(
            (arr - self.mean) / torch.sqrt(self.var + self.epsilon), -1000, 1000
        )


# Source: https://medium.com/@sofeikov/implementing-variational-autoencoders-from-scratch-533782d8eb95
class ConditionalVAE(nn.Module, Sourceable):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self, nail_size: int, hidden_size: int, hammer_size: int, cond_size: int
    ):
        super().__init__()
        self.nail_rms = RunningMeanStd(shape=(nail_size,))
        self.cond_rms = RunningMeanStd(shape=(cond_size,))

        self.encoder = nn.Sequential(
            nn.Linear(nail_size + cond_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hammer_size),
            nn.LeakyReLU(),
        )
        # ia_layer = lambda in_dim, out_dim: torch.nn.Linear(in_dim, out_dim)

        # self.cond_encoder = torch.nn.ModuleList(
        #     [
        #         (
        #             ia_layer(cond_size, op.out_features)
        #             if isinstance(op, torch.nn.Linear)
        #             else torch.nn.Identity()
        #         )
        #         for op in self.encoder
        #     ]
        # )

        self.decoder = nn.Sequential(
            # nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
            nn.Linear(hammer_size + cond_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hammer_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hammer_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hammer_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hammer_size + hidden_size, nail_size),
        )
        # self.decoder = nn.Sequential(
        #     # nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
        #     nn.Linear(hammer_size + cond_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(hammer_size + hidden_size + cond_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(hammer_size + hidden_size + cond_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(hammer_size + hidden_size + cond_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(hammer_size + hidden_size + cond_size, nail_size),
        # )
        # self.cond_decoder = torch.nn.ModuleList(
        #     [
        #         (
        #             ia_layer(cond_size, op.out_features)
        #             if isinstance(op, torch.nn.Linear)
        #             else torch.nn.Identity()
        #         )
        #         for op in self.decoder
        #     ]
        # )

        # Add mu and log_var layers for reparameterization
        self.mu = nn.Linear(hammer_size, hammer_size)
        self.log_var = nn.Linear(hammer_size, hammer_size)
        # self.cond_mu = nn.Linear(cond_size, hammer_size)
        # self.cond_log_var = nn.Linear(cond_size, hammer_size)

        # for e in self.encoder:
        #     for p in e.parameters():
        #         torch.nn.init.zeros_(p)
        #
        # for e in self.decoder:
        #     for p in e.parameters():
        #         torch.nn.init.zeros_(p)
        #
        # for p in self.cond_mu.parameters():
        #     torch.nn.init.zeros_(p)
        #
        # for p in self.cond_log_var.parameters():
        #     torch.nn.init.zeros_(p)

        self.hammer_size = hammer_size
        self.cond_size = cond_size

    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var)
        # Generate random noise using the same shape as std
        eps = torch.randn_like(std)
        # Return the reparameterized sample
        return mu + eps * std

    def reparameterize_zeroNoise(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var)
        # Generate random noise using the same shape as std
        eps = torch.zeros_like(std)
        # Return the reparameterized sample
        return mu + eps * std

    def encode(self, x: torch.Tensor, y: torch.Tensor):
        x = self.nail_rms.normalize(x * 1)
        y = self.cond_rms.normalize(y * 1)
        encoder_out = self.encoder.forward(torch.cat((x, y), dim=-1))
        # for j, op in enumerate(self.encoder):
        #     embed = self.cond_encoder[j]
        #     if isinstance(embed, torch.nn.Identity):
        #         encoder_out = op(encoder_out)
        #     else:
        #         # encoder_out_y = torch.cat((encoder_out, y), dim=-1)
        #         encoder_out = op(encoder_out) + embed(y)
        # for j, op in enumerate(self.encoder):
        #     if isinstance(op, torch.nn.Linear):
        #         encoder_out_y = torch.cat((encoder_out, y), dim=-1)
        #         encoder_out = op(encoder_out_y)
        #     else:
        #         encoder_out = op(encoder_out)
        return encoder_out

    def decode(self, z: torch.Tensor, y: torch.Tensor):
        decoder_out = z * 1
        y = self.cond_rms.normalize(y * 1)
        # for j, op in enumerate(self.decoder):
        #     embed = self.cond_decoder[j]
        #     if isinstance(embed, torch.nn.Identity):
        #         decoder_out = op(decoder_out)
        #     else:
        #         # decoder_out_y = torch.cat((decoder_out, y), dim=-1)
        #         decoder_out = op(decoder_out) + embed(y)
        for j, op in enumerate(self.decoder):
            if isinstance(op, torch.nn.Linear):
                if j > 0:
                    # decoder_out_y = torch.cat((z, decoder_out, y), dim=-1)
                    decoder_out_y = torch.cat((z, decoder_out), dim=-1)
                else:
                    # decoder_out_y = torch.cat((decoder_out, y), dim=-1)
                    decoder_out_y = torch.cat((decoder_out, y), dim=-1)
                decoder_out = op(decoder_out_y)
            else:
                decoder_out = op(decoder_out)
        return decoder_out

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, train_yes: bool
    ) -> (torch.Tensor, torch.Tensor):
        # """
        # Conditioned on `x` and `y`, the encoder enables sampling the embeddings `z`.
        # The decoder is conditioned on `z` and `y` to generate the output.
        # """
        with torch.no_grad():
            if train_yes:
                self.nail_rms.update(x)
                self.cond_rms.update(y)
            # x = self.nail_rms.normalize(x * 1)
            # y = self.cond_rms.normalize(y * 1)

        # Pass the input through the encoder
        encoded = self.encode(x, y)
        # Compute the mean and log variance vectors
        # encoded_y = torch.cat((encoded, y), dim=-1)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        # mu = self.mu(encoded_y)
        # log_var = self.log_var(encoded_y)
        # mu = self.mu(encoded) + self.cond_mu(y)
        # log_var = self.log_var(encoded) + self.cond_log_var(y)
        # Reparameterize the latent variable
        z = self.reparameterize(mu, log_var)
        # Pass the latent variable through the decoder
        decoded = self.decode(z, y)
        return z, decoded, mu, log_var

    def sample(self, y: torch.Tensor) -> torch.Tensor:
        y = self.cond_rms.normalize(y * 1)
        with torch.no_grad():
            # Generate random noise
            z = torch.randn(1, self.hammer_size)
            zy = torch.cat((z, y), dim=-1)
            # Pass the noise through the decoder to generate samples
            samples = self.decoder(z)
        # Return the generated samples
        return samples


class ThroughDataset(Dataset):
    """
    Sacrifice some readability to make life easier.
    Whatever input array/argument tensor provided will be the output for dataset.
    """

    def __init__(self, *args):
        self.args = args
        for a1, a2 in zip(self.args, self.args[1:]):
            assert a1.shape[0] == a2.shape[0]

    def __getitem__(self, index):
        indexed = tuple(torch.as_tensor(a[index]) for a in self.args)
        return indexed

    def __len__(self):
        return self.args[0].shape[0]


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

    # pkl_file_path = f"{proj_dir}/data/nami/torchready/torchready_v2.pkl"
    pkl_file_path = f"{proj_dir}/data/nami/torchready/torchready_v3.pkl"
    data = torch.load(pkl_file_path)
    xs = data["rb_rot_sixd"].reshape(-1, 24, 6)[:, 1:]
    xs = xs.reshape(xs.shape[0], -1)
    xs = torch.tensor(xs, dtype=torch.float)
    ys = np.concatenate(
        [
            data["rb_pos"].reshape(-1, 24, 3)[:, 0, -1, None],  # z position
            data["rb_rot_sixd"].reshape(-1, 24, 6)[:, 0],
            data["rb_vel"].reshape(-1, 24, 3)[:, 0],
            data["rb_ang"].reshape(-1, 24, 3)[:, 0],
        ],
        axis=-1,
    )
    ys = torch.tensor(ys, dtype=torch.float)

    n_train = int(xs.shape[0] * 0.90)
    n_valid = xs.shape[0] - n_train

    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    train_idxs = np.random.choice(xs.shape[0], n_train, replace=False)
    valid_idxs = np.setdiff1d(np.arange(xs.shape[0]), train_idxs)
    xs_train = xs[train_idxs] * 1.0
    ys_train = ys[train_idxs] * 1.0
    print(valid_idxs.shape)
    xs_valid = xs[valid_idxs[: int(valid_idxs.shape[0] / 4)]] * 1.0
    ys_valid = ys[valid_idxs[: int(valid_idxs.shape[0] / 4)]] * 1.0

    # print(torch.max(xs_train, dim = -1))
    xs_train = xs_train.reshape(xs_train.shape[0], -1)
    xs_valid = xs_valid.reshape(xs_valid.shape[0], -1)
    init_lr = 3e-4

    if args.checkpoint_path is None:
        model = ConditionalVAE(xs_valid.shape[-1], 256, args.latent_size, ys.shape[-1])
        model = model.cuda()
        model.nail_rms = model.nail_rms.cuda()
        model.cond_rms = model.cond_rms.cuda()
        model.get_src(0)
        optimizer = Adam(model.parameters(), init_lr)
        global_steps_elapsed = 0
        epochs_elapsed = 0
    else:
        model_dict = torch.load(args.checkpoint_path)
        for line in model_dict["imports"].split("\n"):
            exec(line)
        exec(model_dict["model_src"])
        model = eval(model_dict["model_cls_name"])(
            *model_dict["model_args"], **model_dict["model_kwargs"]
        )
        model.load_state_dict(model_dict["model_state_dict"])
        model = model.to("cuda")
        optimizer = Adam(model.parameters(), init_lr)
        optimizer.load_state_dict(model_dict["optimizer_state_dict"])
        global_steps_elapsed = model_dict["global_steps_elapsed"]
        epochs_elapsed = model_dict["epochs_elapsed"]

    save_every = 1
    # save_every = 10
    # n_total_steps = int(1e7)
    n_total_steps = int(2.5e6)
    pbar = tqdm(total=n_total_steps)
    while global_steps_elapsed <= n_total_steps:

        if epochs_elapsed % save_every == 0 or global_steps_elapsed >= n_total_steps:
            d = {
                "imports": get_imports_as_string(__file__),
                "model_src": model.get_src(0),
                "model_cls_name": model.__class__.__name__,
                "model_args": model.args,
                "model_kwargs": model.kwargs,
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
            ys_valid = ys_valid.cuda()
            z, decoded, mu, log_var = model.forward(xs_valid, ys_valid, False)
            KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            MSE = torch.nn.functional.mse_loss(xs_valid, decoded)
            loss = MSE + KLD * args.kld_weight

            writer.add_scalar("valid/MSEloss", MSE.item(), global_steps_elapsed)
            writer.add_scalar("valid/KLDloss", KLD.item(), global_steps_elapsed)

            # for j in range(10):
            #     writer.add_scalar(
            #         f"valid/mean{j}", torch.mean(z[:, j].abs()), global_steps_elapsed
            #     )
            #     writer.add_scalar(
            #         f"valid/std{j}", torch.std(z[:, j]), global_steps_elapsed
            #     )

        logger.info(
            f"Epoch {epochs_elapsed}:\tMSE: {MSE.item():.3f}\tKLD: {KLD.item():.3f}"
        )

        # lr = init_lr
        lr = init_lr * (1 - (min(1, global_steps_elapsed / n_total_steps)))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if global_steps_elapsed < n_total_steps:
            dataset = ThroughDataset(
                xs_train.cpu().detach().numpy(), ys_train.cpu().detach().numpy()
            )
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            # dataloader = DataLoader(dataset, batch_size=4096, shuffle=False)
            # pbar = tqdm(total=len(dataset))

            for i, (x, y) in enumerate(dataloader):
                x = x.pin_memory().to("cuda", non_blocking=True, dtype=torch.float)
                y = y.pin_memory().to("cuda", non_blocking=True, dtype=torch.float)

                z, decoded, mu, log_var = model(x, y, True)

                # Compute the loss and perform backpropagation
                # KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

                loss = torch.nn.functional.mse_loss(x, decoded) + args.kld_weight * KLD
                # loss += 1e-2 * torch.mean(encoded ** 2)
                # loss_function(decoded, x, mu, log_var, xs_train.shape[-1])#
                optimizer.zero_grad()
                loss.backward()
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
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--override_init_yes", action="store_true")
    parser.add_argument("--debug_yes", action="store_true")
    args = parser.parse_args()

    main()
