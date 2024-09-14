import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn


class RunningMeanStd(nn.Module):
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


class VAE(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_size: int,
        hidden_size: int,
        hammer_size: int,
        out_size: int,
    ):
        super().__init__()
        self.nail_rms = RunningMeanStd(shape=(nail_size,))

        self.encoder = nn.Sequential(
            nn.Linear(nail_size, hidden_size),
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

        self.decoder = nn.Sequential(
            # nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
            nn.Linear(hammer_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hammer_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hammer_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hammer_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hammer_size + hidden_size, out_size),
        )

        # Add mu and log_var layers for reparameterization
        self.mu = nn.Linear(hammer_size, hammer_size)
        self.log_var = nn.Linear(hammer_size, hammer_size)

        self.hammer_size = hammer_size

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

    def encode(self, x: torch.Tensor):
        x = self.nail_rms.normalize(x * 1)
        encoder_out = self.encoder.forward(x)

        return encoder_out

    def decode(self, z: torch.Tensor):
        decoder_out = z * 1

        for j, op in enumerate(self.decoder):
            if isinstance(op, torch.nn.Linear):
                if j > 0:
                    # decoder_out_y = torch.cat((z, decoder_out, y), dim=-1)
                    decoder_out_y = torch.cat((z, decoder_out), dim=-1)
                else:
                    # decoder_out_y = torch.cat((decoder_out, y), dim=-1)
                    decoder_out_y = torch.cat((decoder_out), dim=-1)
                decoder_out = op(decoder_out_y)
            else:
                decoder_out = op(decoder_out)
        return decoder_out

    def forward(self, x: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        # """
        # Conditioned on `x` and `y`, the encoder enables sampling the embeddings `z`.
        # The decoder is conditioned on `z` and `y` to generate the output.
        # """
        with torch.no_grad():
            if train_yes:
                self.nail_rms.update(x)
            # x = self.nail_rms.normalize(x * 1)
            # y = self.cond_rms.normalize(y * 1)

        # Pass the input through the encoder
        encoded = self.encode(x)
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
        decoded = self.decode(z)
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


class PartWiseVAE(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_sizes, #list of ints
        hidden_sizes, #list of ints
        hammer_sizes, #list of ints
        chains_indecies,
    ):
        super().__init__()
        

        self.nail_rmses = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mus = nn.ModuleList()
        self.log_vars = nn.ModuleList()

        for chain_idx, chain in enumerate(chains_indecies):
            nail_size = nail_sizes[chain_idx]
            hidden_size = hidden_sizes[chain_idx]
            hammer_size = hammer_sizes[chain_idx]

            self.nail_rmses.append(RunningMeanStd(shape=(nail_size,)))

            self.encoders.append(nn.Sequential(
                nn.Linear(nail_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hammer_size),
                nn.LeakyReLU(),
            ))

            self.decoders.append(nn.Sequential(
                # nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
                nn.Linear(hammer_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, nail_size),
            ))

            # Add mu and log_var layers for reparameterization
            self.mus.append(nn.Linear(hammer_size, hammer_size))
            self.log_vars.append(nn.Linear(hammer_size, hammer_size))

        self.chains_indecies = chains_indecies
        self.hammer_size = hammer_sizes

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

    def encode(self, x: torch.Tensor, encoder_idx):
        x = self.nail_rmses[encoder_idx].normalize(x * 1)
        encoder_out = self.encoders[encoder_idx].forward(x)

        return encoder_out

    def decode(self, z: torch.Tensor, chain_idx):
        decoder_out = z * 1

        for j, op in enumerate(self.decoders[chain_idx]):
            if isinstance(op, torch.nn.Linear):
                if j > 0:
                    # decoder_out_y = torch.cat((z, decoder_out, y), dim=-1)
                    decoder_out_y = torch.cat((z, decoder_out), dim=-1)
                else:
                    # decoder_out_y = torch.cat((decoder_out, y), dim=-1)
                    decoder_out_y = decoder_out
                decoder_out = op(decoder_out_y)
            else:
                decoder_out = op(decoder_out)
        return decoder_out
    
    def eval_decode(self, z):
        # z = z.reshape(z.shape[0], -1, self.hammer_size)
        # decodeds=[]
        # for chain_idx, chain in enumerate(self.chains_indecies):
        #     chain_z = z[:, chain] * 1
        #     decodeds.append(self.decode(chain_z, chain_idx))
        # return torch.concatenate(decodeds, axis=-1)
        current_start = 0
        decodeds=[]
        for index, hs in enumerate(self.hammer_size):
            chain_z = z[:, current_start: current_start + hs] * 1
            current_start += hs
            decodeds.append(self.cvae.decode(chain_z, index))
        cvae_decoded = torch.concatenate(decodeds, axis=-1)
        return cvae_decoded


    def forward(self, x: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        # """
        # Conditioned on `x` and `y`, the encoder enables sampling the embeddings `z`.
        # The decoder is conditioned on `z` and `y` to generate the output.
        # """
        mus=[]
        log_vars=[]
        zs =[]
        decodeds=[]
        for chain_idx, chain in enumerate(self.chains_indecies):
            chain_x = x[:, chain].reshape(x.shape[0], -1)
            
            with torch.no_grad():
                if train_yes:
                    self.nail_rmses[chain_idx].update(chain_x)
                # x = self.nail_rms.normalize(x * 1)
                # y = self.cond_rms.normalize(y * 1)

            # Pass the input through the encoder
            encoded = self.encode(chain_x, chain_idx)
            
            # Compute the mean and log variance vectors
            # encoded_y = torch.cat((encoded, y), dim=-1)
            mus.append(self.mus[chain_idx](encoded))
            log_vars.append(self.log_vars[chain_idx](encoded))
            # mu = self.mu(encoded_y)
            # log_var = self.log_var(encoded_y)
            # mu = self.mu(encoded) + self.cond_mu(y)
            # log_var = self.log_var(encoded) + self.cond_log_var(y)
            # Reparameterize the latent variable
            zs.append(self.reparameterize(mus[chain_idx], log_vars[chain_idx]))
            # Pass the latent variable through the decoder
            decodeds.append(self.decode(zs[chain_idx], chain_idx))

        z = torch.concatenate(zs, axis=-1)
        decoded = torch.concatenate(decodeds, axis=-1)
        mu = torch.concatenate(mus, axis=-1)
        log_var = torch.concatenate(log_vars, axis=-1)

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

# Source: https://medium.com/@sofeikov/implementing-variational-autoencoders-from-scratch-533782d8eb95
class ConditionalVAE(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_size: int,
        hidden_size: int,
        hammer_size: int,
        cond_size: int,
        out_size: int,
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
            nn.Linear(hammer_size + hidden_size, out_size),
        )
        # Add mu and log_var layers for reparameterization
        self.mu = nn.Linear(hammer_size, hammer_size)
        self.log_var = nn.Linear(hammer_size, hammer_size)

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
        return encoder_out

    def decode(self, z: torch.Tensor, y: torch.Tensor):
        decoder_out = z * 1
        y = self.cond_rms.normalize(y * 1)

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

class ConditionalVAE2(ConditionalVAE):
    def decode(self, z: torch.Tensor, y: torch.Tensor):
        decoder_out = z * 1
        y = self.cond_rms.normalize(y * 1)
        y = y + torch.rand_like(y, device='cuda')
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

class ConditionalPartWiseVAE(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_sizes, #list of ints
        cond_sizes, #list of ints
        hidden_sizes, #list of ints
        hammer_sizes, #list of ints
        chains_indecies,
    ):
        super().__init__()
        

        self.nail_rmses = nn.ModuleList()
        self.cond_rmses = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mus = nn.ModuleList()
        self.log_vars = nn.ModuleList()

        for chain_idx, chain in enumerate(chains_indecies):
            nail_size = nail_sizes[chain_idx]
            cond_size = cond_sizes[chain_idx]
            hidden_size = hidden_sizes[chain_idx]
            hammer_size = hammer_sizes[chain_idx]

            self.nail_rmses.append(RunningMeanStd(shape=(nail_size,)))
            self.cond_rmses.append(RunningMeanStd(shape=(cond_size,)))


            self.encoders.append(nn.Sequential(
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
            ))

            self.decoders.append(nn.Sequential(
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
            ))

            # Add mu and log_var layers for reparameterization
            self.mus.append(nn.Linear(hammer_size, hammer_size))
            self.log_vars.append(nn.Linear(hammer_size, hammer_size))

        self.chains_indecies = chains_indecies
        self.hammer_size = hammer_sizes

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

    def encode(self, x: torch.Tensor,  y: torch.Tensor, encoder_idx):
        x = self.nail_rmses[encoder_idx].normalize(x * 1)
        y = self.cond_rmses[encoder_idx].normalize(y * 1)
        encoder_out = self.encoders[encoder_idx].forward(torch.cat((x, y), dim=-1))

        return encoder_out

    def decode(self, z: torch.Tensor, y: torch.Tensor, chain_idx):
        decoder_out = z * 1
        y = self.cond_rmses[chain_idx].normalize(y * 1)

        for j, op in enumerate(self.decoders[chain_idx]):
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
    
    def eval_decode(self, z):
        # z = z.reshape(z.shape[0], -1, self.hammer_size)
        # decodeds=[]
        # for chain_idx, chain in enumerate(self.chains_indecies):
        #     chain_z = z[:, chain] * 1
        #     decodeds.append(self.decode(chain_z, chain_idx))
        # return torch.concatenate(decodeds, axis=-1)
        current_start = 0
        decodeds=[]
        for index, hs in enumerate(self.hammer_size):
            chain_z = z[:, current_start: current_start + hs] * 1
            current_start += hs
            decodeds.append(self.cvae.decode(chain_z, index))
        cvae_decoded = torch.concatenate(decodeds, axis=-1)
        return cvae_decoded


    def forward(self, x: torch.Tensor,y_lower: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        # """
        # Conditioned on `x` and `y`, the encoder enables sampling the embeddings `z`.
        # The decoder is conditioned on `z` and `y` to generate the output.
        # """
        mus=[]
        log_vars=[]
        zs =[]
        decodeds=[]
        for chain_idx, chain in enumerate(self.chains_indecies):
            chain_x = x[:, chain].reshape(x.shape[0], -1)
            if(chain_idx == 0 ):
                y = y_lower
            else:
                y = decodeds[0] # condition on the lower body

            
            with torch.no_grad():
                if train_yes:
                    self.nail_rmses[chain_idx].update(chain_x)
                    self.cond_rmses[chain_idx].update(y)


            # Pass the input through the encoder
            encoded = self.encode(chain_x,y, chain_idx)

            mus.append(self.mus[chain_idx](encoded))
            log_vars.append(self.log_vars[chain_idx](encoded))

            zs.append(self.reparameterize(mus[chain_idx], log_vars[chain_idx]))
            # Pass the latent variable through the decoder
            decodeds.append(self.decode(zs[chain_idx],y, chain_idx))
            


        z = torch.concatenate(zs, axis=-1)
        decoded = torch.concatenate(decodeds, axis=-1)
        mu = torch.concatenate(mus, axis=-1)
        log_var = torch.concatenate(log_vars, axis=-1)

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
    




class ARConditionalPartWiseVAE(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_sizes, #list of ints
        cond_sizes, #list of ints
        hidden_sizes, #list of ints
        hammer_sizes, #list of ints
        chains_indecies,
        out_sizes,
    ):
        super().__init__()
        

        self.nail_rmses = nn.ModuleList()
        self.cond_rmses = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mus = nn.ModuleList()
        self.log_vars = nn.ModuleList()

        for chain_idx, chain in enumerate(chains_indecies):
            nail_size = nail_sizes[chain_idx]
            cond_size = cond_sizes[chain_idx]
            hidden_size = hidden_sizes[chain_idx]
            hammer_size = hammer_sizes[chain_idx]
            out_size = out_sizes[chain_idx]

            self.nail_rmses.append(RunningMeanStd(shape=(nail_size,)))
            self.cond_rmses.append(RunningMeanStd(shape=(cond_size,)))


            self.encoders.append(nn.Sequential(
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
            ))

            self.decoders.append(nn.Sequential(
                # nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
                nn.Linear(hammer_size + cond_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, out_size),
            ))

            # lower_input_projection_layer = nn.Sequential(
            #     nn.Dropout(0.5),
            #     nn.Linear(lower_size, lower_hidden_size),
            #     # nn.Dropout(0.5),
            # )

            # upper_decoder = nn.Sequential(
            #     nn.Linear(hidden_size + upper_size + lower_hidden_size, hidden_size),
            #     nn.ReLU(),
            # )

            # lower_dropout = nn.Dropout(0.5)
            # upper_decoder = nn.Sequential(
            #     nn.Linear(hidden_size + upper_size + lower_size, hidden_size)
            # )






            # Add mu and log_var layers for reparameterization
            self.mus.append(nn.Linear(hammer_size, hammer_size))
            self.log_vars.append(nn.Linear(hammer_size, hammer_size))

        self.chains_indecies = chains_indecies
        self.hammer_size = hammer_sizes

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

    def encode(self, x: torch.Tensor,  y: torch.Tensor, encoder_idx):
        x = self.nail_rmses[encoder_idx].normalize(x * 1)
        y = self.cond_rmses[encoder_idx].normalize(y * 1)
        encoder_out = self.encoders[encoder_idx].forward(torch.cat((x, y), dim=-1))

        return encoder_out

    def decode(self, z: torch.Tensor, y: torch.Tensor, chain_idx):
        decoder_out = z * 1
        y = self.cond_rmses[chain_idx].normalize(y * 1)

        for j, op in enumerate(self.decoders[chain_idx]):
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
    
    def eval_decode(self, z):
        # z = z.reshape(z.shape[0], -1, self.hammer_size)
        # decodeds=[]
        # for chain_idx, chain in enumerate(self.chains_indecies):
        #     chain_z = z[:, chain] * 1
        #     decodeds.append(self.decode(chain_z, chain_idx))
        # return torch.concatenate(decodeds, axis=-1)
        current_start = 0
        decodeds=[]
        for index, hs in enumerate(self.hammer_size):
            chain_z = z[:, current_start: current_start + hs] * 1
            current_start += hs
            decodeds.append(self.cvae.decode(chain_z, index))
        cvae_decoded = torch.concatenate(decodeds, axis=-1)
        return cvae_decoded


    def forward(self, x_lower: torch.Tensor,  x_upper: torch.Tensor, y_lower: torch.Tensor,y_upper: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        # """
        # Conditioned on `x` and `y`, the encoder enables sampling the embeddings `z`.
        # The decoder is conditioned on `z` and `y` to generate the output.
        # """
        mus=[]
        log_vars=[]
        zs =[]
        decodeds=[]
        for chain_idx, chain in enumerate(self.chains_indecies):
            #chain_x = x[:, chain].reshape(x.shape[0], -1)
            if(chain_idx == 0 ):
                y = y_lower #this is the prev root + lower body pose
                chain_x = x_lower #lower is based on the root
            else:
                y = torch.concatenate([y_upper, decodeds[0]], dim=-1) # this is the prev upper body pose
                chain_x = x_upper #upper is based on the root + lower body

            
            with torch.no_grad():
                if train_yes:
                    self.nail_rmses[chain_idx].update(chain_x)
                    self.cond_rmses[chain_idx].update(y)


            # Pass the input through the encoder
            encoded = self.encode(chain_x,y, chain_idx)

            mus.append(self.mus[chain_idx](encoded))
            log_vars.append(self.log_vars[chain_idx](encoded))

            zs.append(self.reparameterize(mus[chain_idx], log_vars[chain_idx]))
            # Pass the latent variable through the decoder
            decodeds.append(self.decode(zs[chain_idx],y, chain_idx))
            


        z = torch.concatenate(zs, axis=-1)
        decoded = torch.concatenate(decodeds, axis=-1)
        mu = torch.concatenate(mus, axis=-1)
        log_var = torch.concatenate(log_vars, axis=-1)

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
    



#using dropout only for the lowerbody condition for the upper body
#using dropout before the projection stage
class ARConditionalPartWiseVAE2(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_sizes, #list of ints
        cond_sizes, #list of ints
        hidden_sizes, #list of ints
        hammer_sizes, #list of ints
        chains_indecies,
        out_sizes,
    ):
        super().__init__()
        

        self.nail_rmses = nn.ModuleList()
        self.cond_rmses = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mus = nn.ModuleList()
        self.log_vars = nn.ModuleList()

        for chain_idx, chain in enumerate(chains_indecies):
            nail_size = nail_sizes[chain_idx]
            cond_size = cond_sizes[chain_idx]
            hidden_size = hidden_sizes[chain_idx]
            hammer_size = hammer_sizes[chain_idx]
            out_size = out_sizes[chain_idx]

            self.nail_rmses.append(RunningMeanStd(shape=(nail_size,)))
            self.cond_rmses.append(RunningMeanStd(shape=(cond_size,)))


            self.encoders.append(nn.Sequential(
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
            ))

            self.decoders.append(nn.Sequential(
                # nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
                nn.Linear(hammer_size + cond_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, out_size),
            ))

            # lower_input_projection_layer = nn.Sequential(
            #     nn.Dropout(0.5),
            #     nn.Linear(lower_size, lower_hidden_size),
            #     # nn.Dropout(0.5),
            # )

            # upper_decoder = nn.Sequential(
            #     nn.Linear(hidden_size + upper_size + lower_hidden_size, hidden_size),
            #     nn.ReLU(),
            # )

            self.lower_dropout = nn.Dropout(0.5)
            # upper_decoder = nn.Sequential(
            #     nn.Linear(hidden_size + upper_size + lower_size, hidden_size)
            # )






            # Add mu and log_var layers for reparameterization
            self.mus.append(nn.Linear(hammer_size, hammer_size))
            self.log_vars.append(nn.Linear(hammer_size, hammer_size))

        self.chains_indecies = chains_indecies
        self.hammer_size = hammer_sizes

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

    def encode(self, x: torch.Tensor,  y: torch.Tensor, encoder_idx):
        x = self.nail_rmses[encoder_idx].normalize(x * 1)
        y = self.cond_rmses[encoder_idx].normalize(y * 1)
        encoder_out = self.encoders[encoder_idx].forward(torch.cat((x, y), dim=-1))

        return encoder_out

    def decode(self, z: torch.Tensor, y: torch.Tensor, chain_idx):
        decoder_out = z * 1
        y = self.cond_rmses[chain_idx].normalize(y * 1)

        for j, op in enumerate(self.decoders[chain_idx]):
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
    
    def eval_decode(self, z):
        # z = z.reshape(z.shape[0], -1, self.hammer_size)
        # decodeds=[]
        # for chain_idx, chain in enumerate(self.chains_indecies):
        #     chain_z = z[:, chain] * 1
        #     decodeds.append(self.decode(chain_z, chain_idx))
        # return torch.concatenate(decodeds, axis=-1)
        current_start = 0
        decodeds=[]
        for index, hs in enumerate(self.hammer_size):
            chain_z = z[:, current_start: current_start + hs] * 1
            current_start += hs
            decodeds.append(self.cvae.decode(chain_z, index))
        cvae_decoded = torch.concatenate(decodeds, axis=-1)
        return cvae_decoded


    def forward(self, x_lower: torch.Tensor,  x_upper: torch.Tensor, y_lower: torch.Tensor,y_upper: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        # """
        # Conditioned on `x` and `y`, the encoder enables sampling the embeddings `z`.
        # The decoder is conditioned on `z` and `y` to generate the output.
        # """
        mus=[]
        log_vars=[]
        zs =[]
        decodeds=[]
        for chain_idx, chain in enumerate(self.chains_indecies):
            #chain_x = x[:, chain].reshape(x.shape[0], -1)
            if(chain_idx == 0 ):
                y = y_lower #this is the prev root + lower body pose
                chain_x = x_lower #lower is based on the root
            else:
                lower_deocoded = self.lower_dropout(decodeds[0])
                y = torch.concatenate([y_upper, lower_deocoded], dim=-1) # this is the prev upper body pose
                chain_x = x_upper #upper is based on the root + lower body

            
            with torch.no_grad():
                if train_yes:
                    self.nail_rmses[chain_idx].update(chain_x)
                    self.cond_rmses[chain_idx].update(y)


            # Pass the input through the encoder
            encoded = self.encode(chain_x,y, chain_idx)

            mus.append(self.mus[chain_idx](encoded))
            log_vars.append(self.log_vars[chain_idx](encoded))

            zs.append(self.reparameterize(mus[chain_idx], log_vars[chain_idx]))
            # Pass the latent variable through the decoder
            decodeds.append(self.decode(zs[chain_idx],y, chain_idx))
            


        z = torch.concatenate(zs, axis=-1)
        decoded = torch.concatenate(decodeds, axis=-1)
        mu = torch.concatenate(mus, axis=-1)
        log_var = torch.concatenate(log_vars, axis=-1)

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
    

#using dropout only for the lowerbody condition for the upper body
#using dropout after the projection stage
class ARConditionalPartWiseVAE3(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_sizes, #list of ints
        cond_sizes, #list of ints
        hidden_sizes, #list of ints
        hammer_sizes, #list of ints
        chains_indecies,
        out_sizes,
    ):
        super().__init__()
        

        self.nail_rmses = nn.ModuleList()
        self.cond_rmses = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mus = nn.ModuleList()
        self.log_vars = nn.ModuleList()

        for chain_idx, chain in enumerate(chains_indecies):
            nail_size = nail_sizes[chain_idx]
            cond_size = cond_sizes[chain_idx]
            hidden_size = hidden_sizes[chain_idx]
            hammer_size = hammer_sizes[chain_idx]
            out_size = out_sizes[chain_idx]

            self.nail_rmses.append(RunningMeanStd(shape=(nail_size,)))
            self.cond_rmses.append(RunningMeanStd(shape=(cond_size,)))


            self.encoders.append(nn.Sequential(
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
            ))

            self.decoders.append(nn.Sequential(
                # nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
                nn.Linear(hammer_size + cond_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, out_size),
            ))

            # lower_input_projection_layer = nn.Sequential(
            #     nn.Dropout(0.5),
            #     nn.Linear(lower_size, lower_hidden_size),
            #     # nn.Dropout(0.5),
            # )

            # upper_decoder = nn.Sequential(
            #     nn.Linear(hidden_size + upper_size + lower_hidden_size, hidden_size),
            #     nn.ReLU(),
            # )

            self.lower_dropout = nn.Sequential(
                 nn.Linear(cond_sizes[0], cond_sizes[0]),
                 nn.ReLU(),
                 nn.Dropout(0.5)
             )
            
            
            
            # upper_decoder = nn.Sequential(
            #     nn.Linear(hidden_size + upper_size + lower_size, hidden_size)
            # )






            # Add mu and log_var layers for reparameterization
            self.mus.append(nn.Linear(hammer_size, hammer_size))
            self.log_vars.append(nn.Linear(hammer_size, hammer_size))

        self.chains_indecies = chains_indecies
        self.hammer_size = hammer_sizes

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

    def encode(self, x: torch.Tensor,  y: torch.Tensor, encoder_idx):
        x = self.nail_rmses[encoder_idx].normalize(x * 1)
        y = self.cond_rmses[encoder_idx].normalize(y * 1)
        encoder_out = self.encoders[encoder_idx].forward(torch.cat((x, y), dim=-1))

        return encoder_out

    def decode(self, z: torch.Tensor, y: torch.Tensor, chain_idx):
        decoder_out = z * 1
        y = self.cond_rmses[chain_idx].normalize(y * 1)

        for j, op in enumerate(self.decoders[chain_idx]):
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
    
    def eval_decode(self, z):
        # z = z.reshape(z.shape[0], -1, self.hammer_size)
        # decodeds=[]
        # for chain_idx, chain in enumerate(self.chains_indecies):
        #     chain_z = z[:, chain] * 1
        #     decodeds.append(self.decode(chain_z, chain_idx))
        # return torch.concatenate(decodeds, axis=-1)
        current_start = 0
        decodeds=[]
        for index, hs in enumerate(self.hammer_size):
            chain_z = z[:, current_start: current_start + hs] * 1
            current_start += hs
            decodeds.append(self.cvae.decode(chain_z, index))
        cvae_decoded = torch.concatenate(decodeds, axis=-1)
        return cvae_decoded


    def forward(self, x_lower: torch.Tensor,  x_upper: torch.Tensor, y_lower: torch.Tensor,y_upper: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        # """
        # Conditioned on `x` and `y`, the encoder enables sampling the embeddings `z`.
        # The decoder is conditioned on `z` and `y` to generate the output.
        # """
        mus=[]
        log_vars=[]
        zs =[]
        decodeds=[]
        for chain_idx, chain in enumerate(self.chains_indecies):
            #chain_x = x[:, chain].reshape(x.shape[0], -1)
            if(chain_idx == 0 ):
                y = y_lower #this is the prev root + lower body pose
                chain_x = x_lower #lower is based on the root
            else:
                lower_deocoded = self.lower_dropout(decodeds[0])
                y = torch.concatenate([y_upper, lower_deocoded], dim=-1) # this is the prev upper body pose
                chain_x = x_upper #upper is based on the root + lower body

            
            with torch.no_grad():
                if train_yes:
                    self.nail_rmses[chain_idx].update(chain_x)
                    self.cond_rmses[chain_idx].update(y)


            # Pass the input through the encoder
            encoded = self.encode(chain_x,y, chain_idx)

            mus.append(self.mus[chain_idx](encoded))
            log_vars.append(self.log_vars[chain_idx](encoded))

            zs.append(self.reparameterize(mus[chain_idx], log_vars[chain_idx]))
            # Pass the latent variable through the decoder
            decodeds.append(self.decode(zs[chain_idx],y, chain_idx))
            


        z = torch.concatenate(zs, axis=-1)
        decoded = torch.concatenate(decodeds, axis=-1)
        mu = torch.concatenate(mus, axis=-1)
        log_var = torch.concatenate(log_vars, axis=-1)

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
    

#using dropout only for the lowerbody condition for the upper body
#using dropout before the projection stage
class ARConditionalPartWiseVAE4(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_sizes, #list of ints
        cond_sizes, #list of ints
        hidden_sizes, #list of ints
        hammer_sizes, #list of ints
        chains_indecies,
        out_sizes,
    ):
        super().__init__()
        

        self.nail_rmses = nn.ModuleList()
        self.cond_rmses = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mus = nn.ModuleList()
        self.log_vars = nn.ModuleList()

        for chain_idx, chain in enumerate(chains_indecies):
            nail_size = nail_sizes[chain_idx]
            cond_size = cond_sizes[chain_idx]
            hidden_size = hidden_sizes[chain_idx]
            hammer_size = hammer_sizes[chain_idx]
            out_size = out_sizes[chain_idx]

            self.nail_rmses.append(RunningMeanStd(shape=(nail_size,)))
            self.cond_rmses.append(RunningMeanStd(shape=(cond_size,)))


            self.encoders.append(nn.Sequential(
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
            ))

            self.decoders.append(nn.Sequential(
                # nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
                nn.Linear(hammer_size + cond_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, out_size),
            ))

            # lower_input_projection_layer = nn.Sequential(
            #     nn.Dropout(0.5),
            #     nn.Linear(lower_size, lower_hidden_size),
            #     # nn.Dropout(0.5),
            # )

            # upper_decoder = nn.Sequential(
            #     nn.Linear(hidden_size + upper_size + lower_hidden_size, hidden_size),
            #     nn.ReLU(),
            # )

            self.lower_dropout = nn.Sequential(
                nn.Dropout(0.5),
                 nn.Linear(cond_sizes[0], cond_sizes[0]),
                 nn.ReLU(),
                 
             )
            
            
            
            # upper_decoder = nn.Sequential(
            #     nn.Linear(hidden_size + upper_size + lower_size, hidden_size)
            # )






            # Add mu and log_var layers for reparameterization
            self.mus.append(nn.Linear(hammer_size, hammer_size))
            self.log_vars.append(nn.Linear(hammer_size, hammer_size))

        self.chains_indecies = chains_indecies
        self.hammer_size = hammer_sizes

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

    def encode(self, x: torch.Tensor,  y: torch.Tensor, encoder_idx):
        x = self.nail_rmses[encoder_idx].normalize(x * 1)
        y = self.cond_rmses[encoder_idx].normalize(y * 1)
        encoder_out = self.encoders[encoder_idx].forward(torch.cat((x, y), dim=-1))

        return encoder_out

    def decode(self, z: torch.Tensor, y: torch.Tensor, chain_idx):
        decoder_out = z * 1
        y = self.cond_rmses[chain_idx].normalize(y * 1)

        for j, op in enumerate(self.decoders[chain_idx]):
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
    
    def eval_decode(self, z):
        # z = z.reshape(z.shape[0], -1, self.hammer_size)
        # decodeds=[]
        # for chain_idx, chain in enumerate(self.chains_indecies):
        #     chain_z = z[:, chain] * 1
        #     decodeds.append(self.decode(chain_z, chain_idx))
        # return torch.concatenate(decodeds, axis=-1)
        current_start = 0
        decodeds=[]
        for index, hs in enumerate(self.hammer_size):
            chain_z = z[:, current_start: current_start + hs] * 1
            current_start += hs
            decodeds.append(self.cvae.decode(chain_z, index))
        cvae_decoded = torch.concatenate(decodeds, axis=-1)
        return cvae_decoded


    def forward(self, x_lower: torch.Tensor,  x_upper: torch.Tensor, y_lower: torch.Tensor,y_upper: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        # """
        # Conditioned on `x` and `y`, the encoder enables sampling the embeddings `z`.
        # The decoder is conditioned on `z` and `y` to generate the output.
        # """
        mus=[]
        log_vars=[]
        zs =[]
        decodeds=[]
        for chain_idx, chain in enumerate(self.chains_indecies):
            #chain_x = x[:, chain].reshape(x.shape[0], -1)
            if(chain_idx == 0 ):
                y = y_lower #this is the prev root + lower body pose
                chain_x = x_lower #lower is based on the root
            else:
                lower_deocoded = self.lower_dropout(decodeds[0])
                y = torch.concatenate([y_upper, lower_deocoded], dim=-1) # this is the prev upper body pose
                chain_x = x_upper #upper is based on the root + lower body

            
            with torch.no_grad():
                if train_yes:
                    self.nail_rmses[chain_idx].update(chain_x)
                    self.cond_rmses[chain_idx].update(y)


            # Pass the input through the encoder
            encoded = self.encode(chain_x,y, chain_idx)

            mus.append(self.mus[chain_idx](encoded))
            log_vars.append(self.log_vars[chain_idx](encoded))

            zs.append(self.reparameterize(mus[chain_idx], log_vars[chain_idx]))
            # Pass the latent variable through the decoder
            decodeds.append(self.decode(zs[chain_idx],y, chain_idx))
            


        z = torch.concatenate(zs, axis=-1)
        decoded = torch.concatenate(decodeds, axis=-1)
        mu = torch.concatenate(mus, axis=-1)
        log_var = torch.concatenate(log_vars, axis=-1)

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
    


#using dropout for the lowerbody and upperbody condition for the upper body
#using dropout after the projection stage
class ARConditionalPartWiseVAE5(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_sizes, #list of ints
        cond_sizes, #list of ints
        hidden_sizes, #list of ints
        hammer_sizes, #list of ints
        chains_indecies,
        out_sizes,
    ):
        super().__init__()
        

        self.nail_rmses = nn.ModuleList()
        self.cond_rmses = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mus = nn.ModuleList()
        self.log_vars = nn.ModuleList()

        for chain_idx, chain in enumerate(chains_indecies):
            nail_size = nail_sizes[chain_idx]
            cond_size = cond_sizes[chain_idx]
            hidden_size = hidden_sizes[chain_idx]
            hammer_size = hammer_sizes[chain_idx]
            out_size = out_sizes[chain_idx]

            self.nail_rmses.append(RunningMeanStd(shape=(nail_size,)))
            self.cond_rmses.append(RunningMeanStd(shape=(cond_size,)))


            self.encoders.append(nn.Sequential(
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
            ))
            if(chain_idx == 0):
                self.decoders.append(nn.Sequential(
                    # nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
                    nn.Linear(hammer_size + cond_size, hidden_size),
                    nn.LeakyReLU(),
                    nn.Linear(hammer_size + hidden_size, hidden_size),
                    nn.LeakyReLU(),
                    nn.Linear(hammer_size + hidden_size, hidden_size),
                    nn.LeakyReLU(),
                    nn.Linear(hammer_size + hidden_size, hidden_size),
                    nn.LeakyReLU(),
                    nn.Linear(hammer_size + hidden_size, out_size),
                ))
            else:
                self.decoders.append(nn.Sequential(
                    # nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
                    nn.Linear(hammer_size + cond_size, hidden_size),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(hammer_size + hidden_size, hidden_size),
                    nn.LeakyReLU(),
                    nn.Linear(hammer_size + hidden_size, hidden_size),
                    nn.LeakyReLU(),
                    nn.Linear(hammer_size + hidden_size, hidden_size),
                    nn.LeakyReLU(),
                    nn.Linear(hammer_size + hidden_size, out_size),
                ))

            # lower_input_projection_layer = nn.Sequential(
            #     nn.Dropout(0.5),
            #     nn.Linear(lower_size, lower_hidden_size),
            #     # nn.Dropout(0.5),
            # )

            # upper_decoder = nn.Sequential(
            #     nn.Linear(hidden_size + upper_size + lower_hidden_size, hidden_size),
            #     nn.ReLU(),
            # )

            #self.lower_dropout = nn.Dropout(0.5)
            # upper_decoder = nn.Sequential(
            #     nn.Linear(hidden_size + upper_size + lower_size, hidden_size)
            # )






            # Add mu and log_var layers for reparameterization
            self.mus.append(nn.Linear(hammer_size, hammer_size))
            self.log_vars.append(nn.Linear(hammer_size, hammer_size))

        self.chains_indecies = chains_indecies
        self.hammer_size = hammer_sizes

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

    def encode(self, x: torch.Tensor,  y: torch.Tensor, encoder_idx):
        x = self.nail_rmses[encoder_idx].normalize(x * 1)
        y = self.cond_rmses[encoder_idx].normalize(y * 1)
        encoder_out = self.encoders[encoder_idx].forward(torch.cat((x, y), dim=-1))

        return encoder_out

    def decode(self, z: torch.Tensor, y: torch.Tensor, chain_idx):
        decoder_out = z * 1
        y = self.cond_rmses[chain_idx].normalize(y * 1)

        for j, op in enumerate(self.decoders[chain_idx]):
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
    
    def eval_decode(self, z):
        # z = z.reshape(z.shape[0], -1, self.hammer_size)
        # decodeds=[]
        # for chain_idx, chain in enumerate(self.chains_indecies):
        #     chain_z = z[:, chain] * 1
        #     decodeds.append(self.decode(chain_z, chain_idx))
        # return torch.concatenate(decodeds, axis=-1)
        current_start = 0
        decodeds=[]
        for index, hs in enumerate(self.hammer_size):
            chain_z = z[:, current_start: current_start + hs] * 1
            current_start += hs
            decodeds.append(self.cvae.decode(chain_z, index))
        cvae_decoded = torch.concatenate(decodeds, axis=-1)
        return cvae_decoded


    def forward(self, x_lower: torch.Tensor,  x_upper: torch.Tensor, y_lower: torch.Tensor,y_upper: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        # """
        # Conditioned on `x` and `y`, the encoder enables sampling the embeddings `z`.
        # The decoder is conditioned on `z` and `y` to generate the output.
        # """
        mus=[]
        log_vars=[]
        zs =[]
        decodeds=[]
        for chain_idx, chain in enumerate(self.chains_indecies):
            #chain_x = x[:, chain].reshape(x.shape[0], -1)
            if(chain_idx == 0 ):
                y = y_lower #this is the prev root + lower body pose
                chain_x = x_lower #lower is based on the root
            else:
                #lower_deocoded = self.lower_dropout(decodeds[0])
                y = torch.concatenate([y_upper, decodeds[0]], dim=-1) # this is the prev upper body pose
                chain_x = x_upper #upper is based on the root + lower body

            
            with torch.no_grad():
                if train_yes:
                    self.nail_rmses[chain_idx].update(chain_x)
                    self.cond_rmses[chain_idx].update(y)


            # Pass the input through the encoder
            encoded = self.encode(chain_x,y, chain_idx)

            mus.append(self.mus[chain_idx](encoded))
            log_vars.append(self.log_vars[chain_idx](encoded))

            zs.append(self.reparameterize(mus[chain_idx], log_vars[chain_idx]))
            # Pass the latent variable through the decoder
            decodeds.append(self.decode(zs[chain_idx],y, chain_idx))
            


        z = torch.concatenate(zs, axis=-1)
        decoded = torch.concatenate(decodeds, axis=-1)
        mu = torch.concatenate(mus, axis=-1)
        log_var = torch.concatenate(log_vars, axis=-1)

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
    


#using dropout for the lowerbody and upperbody condition for the upper body
#using dropout before the projection stage
class ARConditionalPartWiseVAE6(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_sizes, #list of ints
        cond_sizes, #list of ints
        hidden_sizes, #list of ints
        hammer_sizes, #list of ints
        chains_indecies,
        out_sizes,
    ):
        super().__init__()
        

        self.nail_rmses = nn.ModuleList()
        self.cond_rmses = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mus = nn.ModuleList()
        self.log_vars = nn.ModuleList()

        for chain_idx, chain in enumerate(chains_indecies):
            nail_size = nail_sizes[chain_idx]
            cond_size = cond_sizes[chain_idx]
            hidden_size = hidden_sizes[chain_idx]
            hammer_size = hammer_sizes[chain_idx]
            out_size = out_sizes[chain_idx]

            self.nail_rmses.append(RunningMeanStd(shape=(nail_size,)))
            self.cond_rmses.append(RunningMeanStd(shape=(cond_size,)))

            self.encoders.append(nn.Sequential(
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
            ))

            self.decoders.append(nn.Sequential(
                # nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
                nn.Linear(hammer_size + cond_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, out_size),
            ))

            self.lower_dropout = nn.Sequential(
                 nn.Linear(cond_sizes[1], cond_sizes[1]),
                 nn.ReLU(),
                 nn.Dropout(0.5)
             )

            # Add mu and log_var layers for reparameterization
            self.mus.append(nn.Linear(hammer_size, hammer_size))
            self.log_vars.append(nn.Linear(hammer_size, hammer_size))

        self.chains_indecies = chains_indecies
        self.hammer_size = hammer_sizes

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

    def encode(self, x: torch.Tensor,  y: torch.Tensor, encoder_idx):
        x = self.nail_rmses[encoder_idx].normalize(x * 1)
        y = self.cond_rmses[encoder_idx].normalize(y * 1)
        encoder_out = self.encoders[encoder_idx].forward(torch.cat((x, y), dim=-1))

        return encoder_out

    def decode(self, z: torch.Tensor, y: torch.Tensor, chain_idx):
        decoder_out = z * 1
        y = self.cond_rmses[chain_idx].normalize(y * 1)

        for j, op in enumerate(self.decoders[chain_idx]):
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
    
    def eval_decode(self, z):
        # z = z.reshape(z.shape[0], -1, self.hammer_size)
        # decodeds=[]
        # for chain_idx, chain in enumerate(self.chains_indecies):
        #     chain_z = z[:, chain] * 1
        #     decodeds.append(self.decode(chain_z, chain_idx))
        # return torch.concatenate(decodeds, axis=-1)
        current_start = 0
        decodeds=[]
        for index, hs in enumerate(self.hammer_size):
            chain_z = z[:, current_start: current_start + hs] * 1
            current_start += hs
            decodeds.append(self.cvae.decode(chain_z, index))
        cvae_decoded = torch.concatenate(decodeds, axis=-1)
        return cvae_decoded


    def forward(self, x_lower: torch.Tensor,  x_upper: torch.Tensor, y_lower: torch.Tensor,y_upper: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        # """
        # Conditioned on `x` and `y`, the encoder enables sampling the embeddings `z`.
        # The decoder is conditioned on `z` and `y` to generate the output.
        # """
        mus=[]
        log_vars=[]
        zs =[]
        decodeds=[]
        for chain_idx, chain in enumerate(self.chains_indecies):
            #chain_x = x[:, chain].reshape(x.shape[0], -1)
            if(chain_idx == 0 ):
                y = y_lower #this is the prev root + lower body pose
                chain_x = x_lower #lower is based on the root
            else:
                #lower_deocoded = self.lower_dropout(decodeds[0])
                y = torch.concatenate([y_upper, decodeds[0]], dim=-1) # this is the prev upper body pose
                y = self.lower_dropout(y)
                chain_x = x_upper #upper is based on the root + lower body

            
            with torch.no_grad():
                if train_yes:
                    self.nail_rmses[chain_idx].update(chain_x)
                    self.cond_rmses[chain_idx].update(y)


            # Pass the input through the encoder
            encoded = self.encode(chain_x,y, chain_idx)

            mus.append(self.mus[chain_idx](encoded))
            log_vars.append(self.log_vars[chain_idx](encoded))

            zs.append(self.reparameterize(mus[chain_idx], log_vars[chain_idx]))
            # Pass the latent variable through the decoder
            decodeds.append(self.decode(zs[chain_idx],y, chain_idx))
            


        z = torch.concatenate(zs, axis=-1)
        decoded = torch.concatenate(decodeds, axis=-1)
        mu = torch.concatenate(mus, axis=-1)
        log_var = torch.concatenate(log_vars, axis=-1)

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
    

#using dropout for the lowerbody and upperbody condition for the upper body
#using dropout before the projection stage
class ARConditionalPartWiseVAE7(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_sizes, #list of ints
        cond_sizes, #list of ints
        hidden_sizes, #list of ints
        hammer_sizes, #list of ints
        chains_indecies,
        out_sizes,
    ):
        super().__init__()
        

        self.nail_rmses = nn.ModuleList()
        self.cond_rmses = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mus = nn.ModuleList()
        self.log_vars = nn.ModuleList()

        for chain_idx, chain in enumerate(chains_indecies):
            nail_size = nail_sizes[chain_idx]
            cond_size = cond_sizes[chain_idx]
            hidden_size = hidden_sizes[chain_idx]
            hammer_size = hammer_sizes[chain_idx]
            out_size = out_sizes[chain_idx]

            self.nail_rmses.append(RunningMeanStd(shape=(nail_size,)))
            self.cond_rmses.append(RunningMeanStd(shape=(cond_size,)))

            self.encoders.append(nn.Sequential(
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
            ))

            self.decoders.append(nn.Sequential(
                # nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
                nn.Linear(hammer_size + cond_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, out_size),
            ))

            self.lower_dropout =  nn.Dropout(0.5)

            # Add mu and log_var layers for reparameterization
            self.mus.append(nn.Linear(hammer_size, hammer_size))
            self.log_vars.append(nn.Linear(hammer_size, hammer_size))

        self.chains_indecies = chains_indecies
        self.hammer_size = hammer_sizes

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

    def encode(self, x: torch.Tensor,  y: torch.Tensor, encoder_idx):
        x = self.nail_rmses[encoder_idx].normalize(x * 1)
        y = self.cond_rmses[encoder_idx].normalize(y * 1)
        encoder_out = self.encoders[encoder_idx].forward(torch.cat((x, y), dim=-1))

        return encoder_out

    def decode(self, z: torch.Tensor, y: torch.Tensor, chain_idx):
        decoder_out = z * 1
        y = self.cond_rmses[chain_idx].normalize(y * 1)

        for j, op in enumerate(self.decoders[chain_idx]):
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
    
    def eval_decode(self, z):
        # z = z.reshape(z.shape[0], -1, self.hammer_size)
        # decodeds=[]
        # for chain_idx, chain in enumerate(self.chains_indecies):
        #     chain_z = z[:, chain] * 1
        #     decodeds.append(self.decode(chain_z, chain_idx))
        # return torch.concatenate(decodeds, axis=-1)
        current_start = 0
        decodeds=[]
        for index, hs in enumerate(self.hammer_size):
            chain_z = z[:, current_start: current_start + hs] * 1
            current_start += hs
            decodeds.append(self.cvae.decode(chain_z, index))
        cvae_decoded = torch.concatenate(decodeds, axis=-1)
        return cvae_decoded


    def forward(self, x_lower: torch.Tensor,  x_upper: torch.Tensor, y_lower: torch.Tensor,y_upper: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        # """
        # Conditioned on `x` and `y`, the encoder enables sampling the embeddings `z`.
        # The decoder is conditioned on `z` and `y` to generate the output.
        # """
        mus=[]
        log_vars=[]
        zs =[]
        decodeds=[]
        for chain_idx, chain in enumerate(self.chains_indecies):
            #chain_x = x[:, chain].reshape(x.shape[0], -1)
            if(chain_idx == 0 ):
                y = y_lower #this is the prev root + lower body pose
                chain_x = x_lower #lower is based on the root
            else:
                #lower_deocoded = self.lower_dropout(decodeds[0])
                y = torch.concatenate([y_upper, decodeds[0]], dim=-1) # this is the prev upper body pose
                y = self.lower_dropout(y)
                chain_x = x_upper #upper is based on the root + lower body

            
            with torch.no_grad():
                if train_yes:
                    self.nail_rmses[chain_idx].update(chain_x)
                    self.cond_rmses[chain_idx].update(y)


            # Pass the input through the encoder
            encoded = self.encode(chain_x,y, chain_idx)

            mus.append(self.mus[chain_idx](encoded))
            log_vars.append(self.log_vars[chain_idx](encoded))

            zs.append(self.reparameterize(mus[chain_idx], log_vars[chain_idx]))
            # Pass the latent variable through the decoder
            decodeds.append(self.decode(zs[chain_idx],y, chain_idx))
            


        z = torch.concatenate(zs, axis=-1)
        decoded = torch.concatenate(decodeds, axis=-1)
        mu = torch.concatenate(mus, axis=-1)
        log_var = torch.concatenate(log_vars, axis=-1)

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
    

#using dropout for the lowerbody and upperbody condition for the upper body
#using dropout before the projection stage
class ARConditionalPartWiseVAE8(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_sizes, #list of ints
        cond_sizes, #list of ints
        hidden_sizes, #list of ints
        hammer_sizes, #list of ints
        chains_indecies,
        out_sizes,
    ):
        super().__init__()
        

        self.nail_rmses = nn.ModuleList()
        self.cond_rmses = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mus = nn.ModuleList()
        self.log_vars = nn.ModuleList()

        for chain_idx, chain in enumerate(chains_indecies):
            nail_size = nail_sizes[chain_idx]
            cond_size = cond_sizes[chain_idx]
            hidden_size = hidden_sizes[chain_idx]
            hammer_size = hammer_sizes[chain_idx]
            out_size = out_sizes[chain_idx]

            self.nail_rmses.append(RunningMeanStd(shape=(nail_size,)))
            self.cond_rmses.append(RunningMeanStd(shape=(cond_size,)))

            self.encoders.append(nn.Sequential(
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
            ))

            self.decoders.append(nn.Sequential(
                # nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
                nn.Linear(hammer_size + cond_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, out_size),
            ))

            self.lower_dropout = nn.Sequential(
                nn.Dropout(0.5),
                 nn.Linear(cond_sizes[1], cond_sizes[1]),
                 nn.ReLU(),
                 
             )

            # Add mu and log_var layers for reparameterization
            self.mus.append(nn.Linear(hammer_size, hammer_size))
            self.log_vars.append(nn.Linear(hammer_size, hammer_size))

        self.chains_indecies = chains_indecies
        self.hammer_size = hammer_sizes

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

    def encode(self, x: torch.Tensor,  y: torch.Tensor, encoder_idx):
        x = self.nail_rmses[encoder_idx].normalize(x * 1)
        y = self.cond_rmses[encoder_idx].normalize(y * 1)
        encoder_out = self.encoders[encoder_idx].forward(torch.cat((x, y), dim=-1))

        return encoder_out

    def decode(self, z: torch.Tensor, y: torch.Tensor, chain_idx):
        decoder_out = z * 1
        y = self.cond_rmses[chain_idx].normalize(y * 1)

        for j, op in enumerate(self.decoders[chain_idx]):
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
    
    def eval_decode(self, z):
        # z = z.reshape(z.shape[0], -1, self.hammer_size)
        # decodeds=[]
        # for chain_idx, chain in enumerate(self.chains_indecies):
        #     chain_z = z[:, chain] * 1
        #     decodeds.append(self.decode(chain_z, chain_idx))
        # return torch.concatenate(decodeds, axis=-1)
        current_start = 0
        decodeds=[]
        for index, hs in enumerate(self.hammer_size):
            chain_z = z[:, current_start: current_start + hs] * 1
            current_start += hs
            decodeds.append(self.cvae.decode(chain_z, index))
        cvae_decoded = torch.concatenate(decodeds, axis=-1)
        return cvae_decoded


    def forward(self, x_lower: torch.Tensor,  x_upper: torch.Tensor, y_lower: torch.Tensor,y_upper: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        # """
        # Conditioned on `x` and `y`, the encoder enables sampling the embeddings `z`.
        # The decoder is conditioned on `z` and `y` to generate the output.
        # """
        mus=[]
        log_vars=[]
        zs =[]
        decodeds=[]
        for chain_idx, chain in enumerate(self.chains_indecies):
            #chain_x = x[:, chain].reshape(x.shape[0], -1)
            if(chain_idx == 0 ):
                y = y_lower #this is the prev root + lower body pose
                chain_x = x_lower #lower is based on the root
            else:
                #lower_deocoded = self.lower_dropout(decodeds[0])
                y = torch.concatenate([y_upper, decodeds[0]], dim=-1) # this is the prev upper body pose
                y = self.lower_dropout(y)
                chain_x = x_upper #upper is based on the root + lower body

            
            with torch.no_grad():
                if train_yes:
                    self.nail_rmses[chain_idx].update(chain_x)
                    self.cond_rmses[chain_idx].update(y)


            # Pass the input through the encoder
            encoded = self.encode(chain_x,y, chain_idx)

            mus.append(self.mus[chain_idx](encoded))
            log_vars.append(self.log_vars[chain_idx](encoded))

            zs.append(self.reparameterize(mus[chain_idx], log_vars[chain_idx]))
            # Pass the latent variable through the decoder
            decodeds.append(self.decode(zs[chain_idx],y, chain_idx))
            


        z = torch.concatenate(zs, axis=-1)
        decoded = torch.concatenate(decodeds, axis=-1)
        mu = torch.concatenate(mus, axis=-1)
        log_var = torch.concatenate(log_vars, axis=-1)

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
    
class ARConditionalPartWiseVAE7(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_sizes, #list of ints
        cond_sizes, #list of ints
        hidden_sizes, #list of ints
        hammer_sizes, #list of ints
        chains_indecies,
        out_sizes,
    ):
        super().__init__()
        

        self.nail_rmses = nn.ModuleList()
        self.cond_rmses = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mus = nn.ModuleList()
        self.log_vars = nn.ModuleList()

        for chain_idx, chain in enumerate(chains_indecies):
            nail_size = nail_sizes[chain_idx]
            cond_size = cond_sizes[chain_idx]
            hidden_size = hidden_sizes[chain_idx]
            hammer_size = hammer_sizes[chain_idx]
            out_size = out_sizes[chain_idx]

            self.nail_rmses.append(RunningMeanStd(shape=(nail_size,)))
            self.cond_rmses.append(RunningMeanStd(shape=(cond_size,)))

            self.encoders.append(nn.Sequential(
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
            ))

            self.decoders.append(nn.Sequential(
                # nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
                nn.Linear(hammer_size + cond_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, out_size),
            ))

            self.lower_dropout =  nn.Dropout(0.5)

            # Add mu and log_var layers for reparameterization
            self.mus.append(nn.Linear(hammer_size, hammer_size))
            self.log_vars.append(nn.Linear(hammer_size, hammer_size))

        self.chains_indecies = chains_indecies
        self.hammer_size = hammer_sizes

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

    def encode(self, x: torch.Tensor,  y: torch.Tensor, encoder_idx):
        x = self.nail_rmses[encoder_idx].normalize(x * 1)
        y = self.cond_rmses[encoder_idx].normalize(y * 1)
        encoder_out = self.encoders[encoder_idx].forward(torch.cat((x, y), dim=-1))

        return encoder_out

    def decode(self, z: torch.Tensor, y: torch.Tensor, chain_idx):
        decoder_out = z * 1
        y = self.cond_rmses[chain_idx].normalize(y * 1)

        for j, op in enumerate(self.decoders[chain_idx]):
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
    
    def eval_decode(self, z):
        # z = z.reshape(z.shape[0], -1, self.hammer_size)
        # decodeds=[]
        # for chain_idx, chain in enumerate(self.chains_indecies):
        #     chain_z = z[:, chain] * 1
        #     decodeds.append(self.decode(chain_z, chain_idx))
        # return torch.concatenate(decodeds, axis=-1)
        current_start = 0
        decodeds=[]
        for index, hs in enumerate(self.hammer_size):
            chain_z = z[:, current_start: current_start + hs] * 1
            current_start += hs
            decodeds.append(self.cvae.decode(chain_z, index))
        cvae_decoded = torch.concatenate(decodeds, axis=-1)
        return cvae_decoded


    def forward(self, x_lower: torch.Tensor,  x_upper: torch.Tensor, y_lower: torch.Tensor,y_upper: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        # """
        # Conditioned on `x` and `y`, the encoder enables sampling the embeddings `z`.
        # The decoder is conditioned on `z` and `y` to generate the output.
        # """
        mus=[]
        log_vars=[]
        zs =[]
        decodeds=[]
        for chain_idx, chain in enumerate(self.chains_indecies):
            #chain_x = x[:, chain].reshape(x.shape[0], -1)
            if(chain_idx == 0 ):
                y = y_lower #this is the prev root + lower body pose
                chain_x = x_lower #lower is based on the root
            else:
                #lower_deocoded = self.lower_dropout(decodeds[0])
                y = torch.concatenate([y_upper, decodeds[0]], dim=-1) # this is the prev upper body pose
                y = self.lower_dropout(y)
                chain_x = x_upper #upper is based on the root + lower body

            
            with torch.no_grad():
                if train_yes:
                    self.nail_rmses[chain_idx].update(chain_x)
                    self.cond_rmses[chain_idx].update(y)


            # Pass the input through the encoder
            encoded = self.encode(chain_x,y, chain_idx)

            mus.append(self.mus[chain_idx](encoded))
            log_vars.append(self.log_vars[chain_idx](encoded))

            zs.append(self.reparameterize(mus[chain_idx], log_vars[chain_idx]))
            # Pass the latent variable through the decoder
            decodeds.append(self.decode(zs[chain_idx],y, chain_idx))
            


        z = torch.concatenate(zs, axis=-1)
        decoded = torch.concatenate(decodeds, axis=-1)
        mu = torch.concatenate(mus, axis=-1)
        log_var = torch.concatenate(log_vars, axis=-1)

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
    



class ARSeparateConditionalPartWiseVAE(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_sizes, #list of ints
        cond_sizes, #list of ints
        hidden_sizes, #list of ints
        hammer_sizes, #list of ints
        chains_indecies,
        out_sizes,
    ):
        super().__init__()
        

        self.nail_rmses = nn.ModuleList()
        self.cond_rmses = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mus = nn.ModuleList()
        self.log_vars = nn.ModuleList()

        for chain_idx, chain in enumerate(chains_indecies):
            nail_size = nail_sizes[chain_idx]
            cond_size = cond_sizes[chain_idx]
            hidden_size = hidden_sizes[chain_idx]
            hammer_size = hammer_sizes[chain_idx]
            out_size = out_sizes[chain_idx]

            self.nail_rmses.append(RunningMeanStd(shape=(nail_size,)))
            self.cond_rmses.append(RunningMeanStd(shape=(cond_size,)))


            self.encoders.append(nn.Sequential(
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
            ))

            self.decoders.append(nn.Sequential(
                # nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
                nn.Linear(hammer_size + cond_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, out_size),
            ))

            # Add mu and log_var layers for reparameterization
            self.mus.append(nn.Linear(hammer_size, hammer_size))
            self.log_vars.append(nn.Linear(hammer_size, hammer_size))

        self.chains_indecies = chains_indecies
        self.hammer_size = hammer_sizes

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

    def encode(self, x: torch.Tensor,  y: torch.Tensor, encoder_idx):
        x = self.nail_rmses[encoder_idx].normalize(x * 1)
        y = self.cond_rmses[encoder_idx].normalize(y * 1)
        encoder_out = self.encoders[encoder_idx].forward(torch.cat((x, y), dim=-1))

        return encoder_out

    def decode(self, z: torch.Tensor, y: torch.Tensor, chain_idx):
        decoder_out = z * 1
        y = self.cond_rmses[chain_idx].normalize(y * 1)

        for j, op in enumerate(self.decoders[chain_idx]):
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
    
    def eval_decode(self, z):
        # z = z.reshape(z.shape[0], -1, self.hammer_size)
        # decodeds=[]
        # for chain_idx, chain in enumerate(self.chains_indecies):
        #     chain_z = z[:, chain] * 1
        #     decodeds.append(self.decode(chain_z, chain_idx))
        # return torch.concatenate(decodeds, axis=-1)
        current_start = 0
        decodeds=[]
        for index, hs in enumerate(self.hammer_size):
            chain_z = z[:, current_start: current_start + hs] * 1
            current_start += hs
            decodeds.append(self.cvae.decode(chain_z, index))
        cvae_decoded = torch.concatenate(decodeds, axis=-1)
        return cvae_decoded


    def forward(self, x_lower: torch.Tensor,  x_upper: torch.Tensor, y_lower: torch.Tensor,y_upper: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        # """
        # Conditioned on `x` and `y`, the encoder enables sampling the embeddings `z`.
        # The decoder is conditioned on `z` and `y` to generate the output.
        # """
        mus=[]
        log_vars=[]
        zs =[]
        decodeds=[]
        for chain_idx, chain in enumerate(self.chains_indecies):
            #chain_x = x[:, chain].reshape(x.shape[0], -1)
            if(chain_idx == 0 ):
                y = y_lower #this is the prev root + lower body pose
                chain_x = x_lower #lower is based on the root
            else:
                y = y_upper # this is the prev upper body pose
                chain_x = x_upper #upper is based on the root + lower body

            
            with torch.no_grad():
                if train_yes:
                    self.nail_rmses[chain_idx].update(chain_x)
                    self.cond_rmses[chain_idx].update(y)


            # Pass the input through the encoder
            encoded = self.encode(chain_x,y, chain_idx)

            mus.append(self.mus[chain_idx](encoded))
            log_vars.append(self.log_vars[chain_idx](encoded))

            zs.append(self.reparameterize(mus[chain_idx], log_vars[chain_idx]))
            # Pass the latent variable through the decoder
            decodeds.append(self.decode(zs[chain_idx],y, chain_idx))
            


        z = torch.concatenate(zs, axis=-1)
        decoded = torch.concatenate(decodeds, axis=-1)
        mu = torch.concatenate(mus, axis=-1)
        log_var = torch.concatenate(log_vars, axis=-1)

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
    



class ARConditionalSeparateRootPartWiseVAE(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_sizes, #list of ints
        cond_sizes, #list of ints
        hidden_sizes, #list of ints
        hammer_sizes, #list of ints
        chains_indecies,
        out_sizes,
    ):
        super().__init__()
        

        self.nail_rmses = nn.ModuleList()
        self.cond_rmses = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mus = nn.ModuleList()
        self.log_vars = nn.ModuleList()

        for chain_idx, chain in enumerate(chains_indecies):
            nail_size = nail_sizes[chain_idx]
            cond_size = cond_sizes[chain_idx]
            hidden_size = hidden_sizes[chain_idx]
            hammer_size = hammer_sizes[chain_idx]
            out_size = out_sizes[chain_idx]

            self.nail_rmses.append(RunningMeanStd(shape=(nail_size,)))
            self.cond_rmses.append(RunningMeanStd(shape=(cond_size,)))


            self.encoders.append(nn.Sequential(
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
            ))

            self.decoders.append(nn.Sequential(
                # nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
                nn.Linear(hammer_size + cond_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hammer_size + hidden_size, out_size),
            ))

            # Add mu and log_var layers for reparameterization
            self.mus.append(nn.Linear(hammer_size, hammer_size))
            self.log_vars.append(nn.Linear(hammer_size, hammer_size))

        self.chains_indecies = chains_indecies
        self.hammer_size = hammer_sizes

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

    def encode(self, x: torch.Tensor,  y: torch.Tensor, encoder_idx):
        x = self.nail_rmses[encoder_idx].normalize(x * 1)
        y = self.cond_rmses[encoder_idx].normalize(y * 1)
        encoder_out = self.encoders[encoder_idx].forward(torch.cat((x, y), dim=-1))

        return encoder_out

    def decode(self, z: torch.Tensor, y: torch.Tensor, chain_idx):
        decoder_out = z * 1
        y = self.cond_rmses[chain_idx].normalize(y * 1)

        for j, op in enumerate(self.decoders[chain_idx]):
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
    
    def eval_decode(self, z):
        # z = z.reshape(z.shape[0], -1, self.hammer_size)
        # decodeds=[]
        # for chain_idx, chain in enumerate(self.chains_indecies):
        #     chain_z = z[:, chain] * 1
        #     decodeds.append(self.decode(chain_z, chain_idx))
        # return torch.concatenate(decodeds, axis=-1)
        current_start = 0
        decodeds=[]
        for index, hs in enumerate(self.hammer_size):
            chain_z = z[:, current_start: current_start + hs] * 1
            current_start += hs
            decodeds.append(self.cvae.decode(chain_z, index))
        cvae_decoded = torch.concatenate(decodeds, axis=-1)
        return cvae_decoded


    def forward(self,x_root: torch.Tensor, x_lower: torch.Tensor,  x_upper: torch.Tensor, y_root: torch.Tensor,y_lower: torch.Tensor,y_upper: torch.Tensor, train_yes: bool) -> (torch.Tensor, torch.Tensor):
        # """
        # Conditioned on `x` and `y`, the encoder enables sampling the embeddings `z`.
        # The decoder is conditioned on `z` and `y` to generate the output.
        # """
        mus=[]
        log_vars=[]
        zs =[]
        decodeds=[]
        for chain_idx, chain in enumerate(self.chains_indecies):
            #chain_x = x[:, chain].reshape(x.shape[0], -1)
            if(chain_idx == 0 ):
                y = y_root #this is the prev root + lower body pose
                chain_x = x_root #lower is based on the root
            elif(chain_idx == 1):
                y = torch.concatenate([y_lower, decodeds[0]], dim=-1) # this is the prev upper body pose
                chain_x = x_lower #upper is based on the root + lower body
            else:
                y = torch.concatenate([y_upper, decodeds[1]], dim=-1) # this is the prev upper body pose
                chain_x = x_upper #upper is based on the root + lower body

            
            with torch.no_grad():
                if train_yes:
                    self.nail_rmses[chain_idx].update(chain_x)
                    self.cond_rmses[chain_idx].update(y)


            # Pass the input through the encoder
            encoded = self.encode(chain_x,y, chain_idx)

            mus.append(self.mus[chain_idx](encoded))
            log_vars.append(self.log_vars[chain_idx](encoded))

            zs.append(self.reparameterize(mus[chain_idx], log_vars[chain_idx]))
            # Pass the latent variable through the decoder
            decodeds.append(self.decode(zs[chain_idx],y, chain_idx))
            


        z = torch.concatenate(zs, axis=-1)
        decoded = torch.concatenate(decodeds, axis=-1)
        mu = torch.concatenate(mus, axis=-1)
        log_var = torch.concatenate(log_vars, axis=-1)

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
    



class ConditionalWAE(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_size: int,
        hidden_size: int,
        hammer_size: int,
        cond_size: int,
        out_size: int,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(nail_size + cond_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.LeakyReLU(),  # Should apply activation only on the embeddings, not the condition, so this has to be pushed to `forward`
            nn.Linear(hammer_size + cond_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear( hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear( hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear( hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear( hidden_size, out_size),
        )
        
        self.mu = nn.Sequential(nn.LeakyReLU(), nn.Linear(hidden_size, hammer_size))
        self.logstd = nn.Sequential(nn.LeakyReLU(), nn.Linear(hidden_size, hammer_size))

        self.hammer_size = hammer_size
        self.cond_size = cond_size
        self.input_rms = RunningMeanStd(shape=(self.cond_size,))


    def encode(self, x: torch.Tensor, y: torch.Tensor):
        encoder_out = self.encoder.forward(torch.cat((self.input_rms.normalize(x), self.input_rms.normalize(y)), dim=-1))
        return encoder_out

    def decode(self, z: torch.Tensor, y: torch.Tensor):
        #decoder_out = z * 1
        #y = self.input_rms.normalize(y)
        decoder_out = self.decoder.forward(torch.cat((z, self.input_rms.normalize(y)), dim=-1))

        # for j, op in enumerate(self.decoder):
        #     if isinstance(op, torch.nn.Linear):
        #         if j > 1:
        #             # decoder_out_y = torch.cat((z, decoder_out, y), dim=-1)
        #             decoder_out_y = torch.cat((z, decoder_out), dim=-1)
        #         else:
        #             # decoder_out_y = torch.cat((decoder_out, y), dim=-1)
        #             decoder_out_y = torch.cat((decoder_out, y), dim=-1)
        #         decoder_out = op(decoder_out_y)
        #     else:
        #         decoder_out = op(decoder_out)
        return decoder_out

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, train_yes: bool
    ) -> (torch.Tensor, torch.Tensor):

        z = self.encode(x, y)

        mu = self.mu(z)
        logstd = self.logstd(z)

        noise = torch.randn_like(mu)
        z = mu + noise * torch.exp(logstd)
    
        decoded = self.decode(z, y)
        return z, decoded
    

class WAEDisc(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        nail_size: int,
        hidden_size: int,
        hammer_size: int,
        cond_size: int,
        out_size: int,
    ):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(hammer_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            #nn.Sigmoid()
        )

        
        self.hammer_size = hammer_size
        self.cond_size = cond_size
    def forward(self, x):
        x = self.disc(x)
        return x



def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False