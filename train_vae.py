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

    def get_src(self, depth):
        my_src = f"global {self.__class__.__name__}\n" + inspect.getsource(
            self.__class__
        )
        srcs = []
        if depth == 0:
            srcs.append("global Sourceable\n" + inspect.getsource(Sourceable))
        for child_module in self._modules:
            child_module_inst = getattr(self, child_module)
            if isinstance(child_module_inst, Sourceable):
                srcs.append(child_module_inst.get_src(depth + 1))
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

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean * 1
        new_object.var = self.var * 1
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

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
class VAE(nn.Module, Sourceable):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(self, nail_size: int, hidden_size: int, hammer_size: int):
        super().__init__()
        self.rms = RunningMeanStd(shape=(nail_size,))

        self.encoder = nn.Sequential(
            nn.Linear(nail_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hammer_size),
            nn.LeakyReLU(),
        )
        # if args.override_init_yes:
        #     for name, param in self.encoder.named_parameters():
        #         if name.endswith(".bias"):
        #             param.data.fill_(0)
        #         elif name.startswith(
        #             "0"
        #         ):  # The first layer does not have ReLU applied on its input
        #             param.data.normal_(0, 1 / np.sqrt(param.shape[1]))
        self.decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hammer_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, nail_size),
        )
        # for name, param in self.decoder.named_parameters():
        #     if name.endswith(".bias"):
        #         param.data.fill_(0)

        self.test = nn.Sequential(
            nn.Linear(nail_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size, track_running_stats=False),
        )
        self.hammer_size = hammer_size
        # Add mu and log_var layers for reparameterization
        self.mu = nn.Linear(hammer_size, hammer_size)
        self.log_var = nn.Linear(hammer_size, hammer_size)

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

    def forward(self, x: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            if train_yes:
                self.rms.update(x)
            x = self.rms.normalize(x * 1)
        # Pass the input through the encoder
        # test = self.test(x)
        encoded = self.encoder(x)
        # Compute the mean and log variance vectors
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        # Reparameterize the latent variable
        z = self.reparameterize(mu, log_var)
        # Pass the latent variable through the decoder
        decoded = self.decoder(z)
        # Return the encoded output, decoded output, mean, and log variance
        # return encoded, decoded, mu, log_var, test
        return z, decoded, mu, log_var

    def sample(self):
        with torch.no_grad():
            # Generate random noise
            z = torch.randn(1, self.hammer_size)
            # Pass the noise through the decoder to generate samples
            samples = self.decoder(z)
        # Return the generated samples
        return samples

    def map_gaussian_noise(self, noise: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            samples = self.decoder.forward(noise)
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

    pkl_file_path = f"{proj_dir}/data/nami/torchready/torchready_v2.pkl"
    data = torch.load(pkl_file_path)
    xs = data["rb_rot_sixd"]
    xs = torch.tensor(xs, dtype=torch.float)

    n_train = int(xs.shape[0] * 0.90)
    n_valid = xs.shape[0] - n_train

    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    train_idxs = np.random.choice(xs.shape[0], n_train, replace=False)
    valid_idxs = np.setdiff1d(np.arange(xs.shape[0]), train_idxs)
    xs_train = xs[train_idxs] * 1.0
    print(valid_idxs.shape)
    xs_valid = xs[valid_idxs[: int(valid_idxs.shape[0] / 4)]] * 1.0

    # print(torch.max(xs_train, dim = -1))
    xs_train = xs_train.reshape(xs_train.shape[0], -1)
    xs_valid = xs_valid.reshape(xs_valid.shape[0], -1)

    if args.checkpoint_path is None:
        model = VAE(xs_valid.shape[-1], 256, 64)
        model = model.cuda()
        model.rms = model.rms.cuda()
        model.get_src(0)
        optimizer = Adam(model.parameters(), 3e-4)
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
        optimizer = Adam(model.parameters(), 3e-4)
        optimizer.load_state_dict(model_dict["optimizer_state_dict"])

    global_steps_elapsed = 0
    epochs_elapsed = 0
    save_every = 100
    n_total_steps = int(1e5)
    pbar = tqdm(total=n_total_steps)
    # for ste in range(n_total_steps):
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
            z, decoded, mu, log_var = model.forward(xs_valid, False)
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            MSE = torch.nn.functional.mse_loss(xs_valid, decoded)
            loss = MSE + KLD * args.kld_weight
            xs_valid_np = xs_valid.detach().cpu().numpy()
            decoded_np = decoded.detach().cpu().numpy()

            digit_size = 128
            n = 5
            norm = td.Normal(0, 1)
            grid_y = norm.icdf(torch.linspace(0.05, 0.95, n))
            writer.add_scalar("valid/MSEloss", MSE.item(), global_steps_elapsed)
            writer.add_scalar("valid/KLDloss", KLD.item(), global_steps_elapsed)

            for j in range(10):
                writer.add_scalar(
                    f"valid/mean{j}", torch.mean(z[:, j].abs()), global_steps_elapsed
                )
                writer.add_scalar(
                    f"valid/std{j}", torch.std(z[:, j]), global_steps_elapsed
                )

        logger.info(
            f"Epoch {epochs_elapsed}:\tMSE: {MSE.item():.3f}\tKLD: {KLD.item():.3f}"
        )

        if global_steps_elapsed < n_total_steps:
            dataset = ThroughDataset(xs_train.cpu().detach().numpy())
            dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
            # dataloader = DataLoader(dataset, batch_size=4096, shuffle=False)
            # pbar = tqdm(total=len(dataset))

            for i, (x,) in enumerate(dataloader):
                x = x.pin_memory().to("cuda", non_blocking=True, dtype=torch.float)

                z, decoded, mu, log_var = model(x, True)

                # Compute the loss and perform backpropagation
                KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

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
        else:
            break

    pbar.close()
    writer.flush()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--kld_weight", type=float, default=0)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--override_init_yes", action="store_true")
    parser.add_argument("--debug_yes", action="store_true")
    args = parser.parse_args()

    main()
