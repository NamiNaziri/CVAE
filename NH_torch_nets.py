import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from poselib.poselib.skeleton.skeleton3d import SkeletonState
from torch.utils.data import Dataset

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


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float)


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

    def unnormalize(self, arr: torch.Tensor) -> torch.Tensor:
        return arr * torch.sqrt(self.var + self.epsilon) + self.mean


class TransformerDiscrete(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ff_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        vocab_size: int,
        sentence_length: int,
        n_sentences_in_paragraph: int,
        n_paragraphs_in_article: int,
        device: torch.device,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sentence_length = sentence_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_size = ff_size
        self.dropout = dropout
        self.n_sentences_in_paragraph = n_sentences_in_paragraph
        self.n_paragraphs_in_article = n_paragraphs_in_article
        self.device = device

        # emb_size = 1024
        # self.emb = nn.Embedding(vocab_size, emb_size)
        self.emb = nn.Embedding(vocab_size, hidden_size)
        # self.proj = nn.Sequential(
        #     nn.Linear(emb_size, self.hidden_size, bias=True),
        # )

        self.activation = nn.GELU()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            bias=True,
        )
        self.transformer_decoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.deproj = nn.Linear(self.hidden_size, self.vocab_size, bias=True)
        self.paragraph_length = self.n_sentences_in_paragraph * self.sentence_length
        self.article_length = self.paragraph_length * self.n_paragraphs_in_article
        self.ln = nn.LayerNorm(self.hidden_size)

        self.pe = torch.zeros(
            (self.article_length, self.hidden_size), device=self.device
        )
        div_term = torch.exp(
            torch.arange(0, self.hidden_size // 2, 1).float()
            * (-np.log(10000.0) / (self.hidden_size))
        ).to(self.device)
        self.pe[..., 0::2] = torch.sin(
            torch.arange(self.pe.shape[0], device=self.pe.device)[..., None] * div_term
        )
        self.pe[..., 1::2] = torch.cos(
            torch.arange(self.pe.shape[0], device=self.pe.device)[..., None] * div_term
        )

        self.tgt_mask = ~torch.tril(
            torch.ones(
                self.article_length + 1,
                self.article_length + 1,
                dtype=torch.bool,
                device=self.device,
            ),
            0,
        )
        a = torch.ones(
            (
                self.n_paragraphs_in_article,
                self.n_paragraphs_in_article,
                self.sentence_length * self.n_sentences_in_paragraph,
                self.sentence_length * self.n_sentences_in_paragraph,
            ),
            device=self.device,
            dtype=torch.bool,
        )
        for i in range(self.n_paragraphs_in_article):
            a[i, i] = torch.eye(self.sentence_length * self.n_sentences_in_paragraph)
            for j in range(i):
                a[i, j] = False
        a = a.permute(0, 2, 1, 3).reshape(self.article_length, self.article_length)
        self.completion_mask = torch.zeros(
            (self.article_length + 1, self.article_length + 1),
            device=self.device,
            dtype=torch.bool,
        )
        self.completion_mask[1:, 1:] = a
        self.completion_mask = self.completion_mask[1:, :-1]

        self.pseudo_causal_masks = []
        for i in range(self.n_sentences_in_paragraph):
            a = torch.ones(
                (self.article_length, self.article_length),
                device=self.device,
                dtype=torch.bool,
            )
            a = a.reshape(
                self.n_paragraphs_in_article,
                self.n_sentences_in_paragraph,
                self.sentence_length,
                self.n_paragraphs_in_article,
                self.n_sentences_in_paragraph,
                self.sentence_length,
            )
            for j in range(self.n_paragraphs_in_article):
                a[j, :, :, j, i, :] = False
            a = a.reshape(self.article_length, self.article_length)
            pseudo_causal_mask = ~torch.tril(
                torch.ones(
                    self.article_length + 1,
                    self.article_length + 1,
                    dtype=torch.bool,
                    device=self.device,
                ),
                0,
            )
            pseudo_causal_mask[:-1, 1:] = torch.logical_and(
                a, pseudo_causal_mask[:-1, 1:]
            )
            # pseudo_causal_mask = pseudo_causal_mask[:-1, :-1]
            self.pseudo_causal_masks.append(pseudo_causal_mask)
        self.pseudo_causal_masks.append(self.tgt_mask)
        self.pseudo_causal_masks = torch.stack(self.pseudo_causal_masks, dim=0)

        self.rms_root_vxyz = RunningMeanStd(shape=(3,))
        self.rms_root_xyz = RunningMeanStd(shape=(3,))
        self.rms_root_expm = RunningMeanStd(shape=(3,))
        self.rms_upper_expm = RunningMeanStd(shape=(39,))
        self.rms_lower_expm = RunningMeanStd(shape=(30,))

        self.root_vxyz_tokenizer = RVQTokenizer(
            3, self.sentence_length, self.vocab_size, 50, 1, False
        )
        self.root_xyz_tokenizer = RVQTokenizer(
            3, self.sentence_length, self.vocab_size, 50, 1, False
        )
        self.root_expm_tokenizer = RVQTokenizer(
            3, self.sentence_length, self.vocab_size, 50, 1, False
        )
        self.upper_expm_tokenizer = RVQTokenizer(
            39, self.sentence_length, vocab_size, 50, 1, False
        )
        self.lower_expm_tokenizer = RVQTokenizer(
            30, self.sentence_length, vocab_size, 50, 1, False
        )

    def setup(
        self,
        root_vxyz: torch.Tensor,
        root_xyz: torch.Tensor,
        root_expm: torch.Tensor,
        upper_expm: torch.Tensor,
        lower_expm: torch.Tensor,
    ):

        self.rms_root_vxyz.to(device=self.device)
        self.rms_root_vxyz.update(root_vxyz)
        self.root_vxyz_tokenizer.build_codebook(
            self.rms_root_vxyz.normalize(root_vxyz), device=self.device
        )

        self.rms_root_xyz.to(device=self.device)
        self.rms_root_xyz.update(root_xyz)
        self.root_xyz_tokenizer.build_codebook(
            self.rms_root_xyz.normalize(root_xyz), device=self.device
        )

        self.rms_root_expm.to(device=self.device)
        self.rms_root_expm.update(root_expm)
        self.root_expm_tokenizer.build_codebook(
            self.rms_root_expm.normalize(root_expm), device=self.device
        )

        self.rms_upper_expm.to(device=self.device)
        upper_expm = upper_expm.reshape(upper_expm.shape[0], -1)
        self.rms_upper_expm.update(upper_expm)
        self.upper_expm_tokenizer.build_codebook(
            self.rms_upper_expm.normalize(upper_expm), device=self.device
        )

        self.rms_lower_expm.to(device=self.device)
        lower_expm = lower_expm.reshape(lower_expm.shape[0], -1)
        self.rms_lower_expm.update(lower_expm)
        self.lower_expm_tokenizer.build_codebook(
            self.rms_lower_expm.normalize(lower_expm), device=self.device
        )

    def prep_paragraphs(self, xyzexpm: torch.Tensor):
        if xyzexpm.numel() == 0:
            return torch.empty(
                (xyzexpm.shape[0], 0, 4, self.sentence_length),
                dtype=torch.long,
                device=self.device,
            )
        n = xyzexpm.shape[0]
        t = xyzexpm.shape[1]

        root_vxyzs = xyzexpm[:, :, 0]
        root_xyzs = xyzexpm[:, :, 1]
        expm = xyzexpm[:, :, 2:]

        root_expms = expm[:, :, 0]
        upper_expms = expm[:, :, upper_body_idxs]
        lower_expms = expm[:, :, lower_body_idxs]

        root_vxyz_tokens, _ = self.root_vxyz_tokenizer.encode(
            self.rms_root_vxyz.normalize(root_vxyzs.reshape(-1, 3))
        )
        root_vxyz_tokens = root_vxyz_tokens.reshape(n, t, -1)

        root_xyz_tokens, _ = self.root_xyz_tokenizer.encode(
            self.rms_root_xyz.normalize(root_xyzs.reshape(-1, 3))
        )
        root_xyz_tokens = root_xyz_tokens.reshape(n, t, -1)

        root_expm_tokens, _ = self.root_expm_tokenizer.encode(
            self.rms_root_expm.normalize(root_expms.reshape(-1, 3))
        )
        root_expm_tokens = root_expm_tokens.reshape(n, t, -1)

        upper_expm_tokens, _ = self.upper_expm_tokenizer.encode(
            self.rms_upper_expm.normalize(upper_expms.reshape(-1, 39))
        )
        upper_expm_tokens = upper_expm_tokens.reshape(n, t, -1)

        lower_expm_tokens, _ = self.lower_expm_tokenizer.encode(
            self.rms_lower_expm.normalize(lower_expms.reshape(-1, 30))
        )
        lower_expm_tokens = lower_expm_tokens.reshape(n, t, -1)

        paragraphs = torch.stack(
            [
                root_vxyz_tokens,
                root_xyz_tokens,
                root_expm_tokens,
                upper_expm_tokens,
                lower_expm_tokens,
            ],
            dim=2,
        )
        return paragraphs

    def forward(
        self,
        xyzexpm: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        paragraphs = self.prep_paragraphs(xyzexpm)

        xx = self.emb(paragraphs)
        xx = xx.reshape(xx.shape[0], -1, xx.shape[-1])
        tgt = xx + self.pe[None, : xx.shape[1]]
        tgt = torch.cat([torch.zeros_like(xx[:, [0]]), tgt], dim=1)
        # x_hat = self.convert_articles_to_xyzexpms(articles)

        # Causal sampling task loss
        mask_repeated = mask.repeat_interleave(self.paragraph_length, -1)
        # tgt_mask = self.tgt_mask.clone()
        # decoder_out = self.transformer_decoder(
        #     src=tgt[:, :-1],
        #     mask=tgt_mask,
        #     is_causal=True,
        # )
        # decoder_out = self.ln(decoder_out)
        # logits = self.deproj(decoder_out)
        #
        # causal_loss = torch.nn.functional.cross_entropy(
        #     logits[mask_repeated].reshape(-1, logits.shape[-1]),
        #     paragraphs[mask].reshape(-1),
        # )
        # Masked token completion loss: have access up to the current time
        # Note: this is not strictly causal
        # tgt_mask = self.completion_mask
        # decoder_out = self.transformer_decoder(
        #     src=tgt[:, :-1],
        #     mask=tgt_mask,
        #     is_causal=False,
        # )
        # decoder_out = self.ln(decoder_out)
        # logits = self.deproj(decoder_out)
        # mask_completion_loss = torch.nn.functional.cross_entropy(
        #     logits[mask_repeated].reshape(-1, logits.shape[-1]),
        #     paragraphs[mask].reshape(-1),
        # )

        # Pseudo-causal masks
        pseudo_causal_losses = []
        for i in range(self.n_sentences_in_paragraph):
            # for i in range(1):
            # loss_mask = torch.ones(self.article_length, device=self.device, dtype=torch.bool)
            # loss_mask = loss_mask.reshape(-1, 4)
            # loss_mask[:, i] = False
            # loss_mask = loss_mask.reshape(-1)
            # a = self.tgt_mask.clone().reshape(
            #     self.n_paragraphs_in_article,
            #     self.n_sentences_in_paragraph * self.sentence_length,
            #     self.n_paragraphs_in_article,
            #     self.n_sentences_in_paragraph * self.sentence_length,
            # )
            # for j in range(self.n_paragraphs_in_article):
            #     a[j, :, j, i] = False
            #     # a[j, i - 1, j, i] = True
            # a = a.reshape(self.article_length, self.article_length)

            decoder_out = self.transformer_decoder(
                src=tgt[:, :-1],
                mask=self.pseudo_causal_masks[i][:-1, :-1],
                is_causal=False,
            )
            decoder_out = self.ln(decoder_out)
            logits = self.deproj(decoder_out)
            pseudo_causal_loss = torch.nn.functional.cross_entropy(
                logits[mask_repeated].reshape(-1, logits.shape[-1]),
                paragraphs[mask].reshape(-1),
            )
            pseudo_causal_losses.append(pseudo_causal_loss)

        loss = torch.stack(
            [
                # causal_loss,
                # mask_completion_loss,
                torch.stack(pseudo_causal_losses).mean(),
            ]
        ).mean()
        return loss

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), x.view(-1)
        )
        return loss

    def convert_articles_to_xyzexpms(self, articles: torch.Tensor) -> torch.Tensor:
        n = articles.shape[0]
        t = articles.shape[1]
        root_vxyz_tokens = articles[:, :, 0]
        root_xyz_tokens = articles[:, :, 1]
        root_expm_tokens = articles[:, :, 2]
        upper_expm_tokens = articles[:, :, 3]
        lower_expm_tokens = articles[:, :, 4]

        root_vxyz = self.root_vxyz_tokenizer.decode(
            root_vxyz_tokens.reshape(-1, self.sentence_length), device=self.device
        )
        root_vxyz = root_vxyz.reshape(n, t, 1, 3)
        root_vxyz = self.rms_root_vxyz.unnormalize(root_vxyz)

        root_xyz = self.root_xyz_tokenizer.decode(
            root_xyz_tokens.reshape(-1, self.sentence_length), device=self.device
        )
        root_xyz = root_xyz.reshape(n, t, 1, 3)
        root_xyz = self.rms_root_xyz.unnormalize(root_xyz)

        root_expm = self.root_expm_tokenizer.decode(
            root_expm_tokens.reshape(-1, self.sentence_length), device=self.device
        )
        root_expm = root_expm.reshape(n, t, 1, 3)
        root_expm = self.rms_root_expm.unnormalize(root_expm)

        upper_expm = self.upper_expm_tokenizer.decode(
            upper_expm_tokens.reshape(-1, self.sentence_length),
            device=self.device,
        )
        upper_expm = upper_expm.reshape(n, t, 39)
        upper_expm = self.rms_upper_expm.unnormalize(upper_expm)
        upper_expm = upper_expm.reshape(n, t, 13, 3)

        lower_expm = self.lower_expm_tokenizer.decode(
            lower_expm_tokens.reshape(-1, self.sentence_length), device=self.device
        )
        lower_expm = lower_expm.reshape(n, t, 30)
        lower_expm = self.rms_lower_expm.unnormalize(lower_expm)
        lower_expm = lower_expm.reshape(n, t, 10, 3)

        expm = torch.zeros((n, t, 24, 3), device=self.device)
        expm[:, :, [0]] = root_expm
        expm[:, :, upper_body_idxs] = upper_expm
        expm[:, :, lower_body_idxs] = lower_expm

        xyzexpm = torch.cat([root_vxyz, root_xyz, expm], dim=2)
        return xyzexpm

    def generate(
        self,
        xyzexpm: torch.Tensor,
        n_paragraphs: int,
        topk: int = 1,
        temperature: float = 1,
    ) -> torch.Tensor:
        paragraphs = self.prep_paragraphs(xyzexpm)

        xx = self.emb(paragraphs)
        xx = xx.reshape(xx.shape[0], -1, xx.shape[-1])
        tgt = xx + self.pe[None, : xx.shape[1]]
        tgt = torch.cat(
            [
                torch.zeros(
                    (xyzexpm.shape[0], 1, tgt.shape[-1]), device=self.device
                ),  # beginning of article
                tgt,
            ],
            dim=1,
        )

        t = tgt.shape[1] - 1
        words = [paragraphs.reshape(paragraphs.shape[0], -1)]
        # words = []
        for i in range(n_paragraphs):
            for j in range(self.n_sentences_in_paragraph):
                for k in range(self.sentence_length):
                    my_mask = self.tgt_mask[
                        : t + 1,
                        : t + 1,
                    ]
                    decoder_out = self.transformer_decoder(
                        src=tgt,
                        mask=my_mask,
                        is_causal=False,
                    )
                    decoder_out = self.ln(decoder_out)
                    next_logits = self.deproj(decoder_out[:, -1])
                    next_logits /= temperature
                    v, _ = torch.topk(next_logits, min(topk, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -torch.inf
                    # Stochastic
                    next_tok = torch.multinomial(next_logits.softmax(-1), 1)
                    words.append(next_tok)
                    next_emb = self.emb(next_tok)
                    tgt = torch.cat([tgt, next_emb + self.pe[t]], dim=1)
                    t += 1
        words = torch.cat(words, dim=1)
        articles = words.reshape(
            words.shape[0],
            n_paragraphs + xyzexpm.shape[1],
            # n_paragraphs,
            self.n_sentences_in_paragraph,
            self.sentence_length,
        )

        x_hat = self.convert_articles_to_xyzexpms(articles)

        return articles, x_hat

    def regen(
        self,
        xyzexpm: torch.Tensor,
        regen_i: int = 0,
        topk: int = 1,
        temperature: float = 1,
    ) -> torch.Tensor:
        paragraphs = self.prep_paragraphs(xyzexpm)

        xx = self.emb(paragraphs)
        xx = xx.reshape(xx.shape[0], -1, xx.shape[-1])
        tgt = xx + self.pe[None, : xx.shape[1]]
        tgt = torch.cat(
            [
                torch.zeros(
                    (xyzexpm.shape[0], 1, tgt.shape[-1]), device=self.device
                ),  # beginning of article
                tgt,
            ],
            dim=1,
        )

        # Use pseudo causal masking to regenerate the full sentence based on one token
        words = [paragraphs[:, :-1].reshape(paragraphs.shape[0], -1)]
        t = tgt.shape[1] - 1
        tt = t - self.n_sentences_in_paragraph * self.sentence_length + 1
        skip_me = torch.arange(tt + regen_i, tt + regen_i + self.sentence_length)
        # Scrapping tgt just to make sure
        # tmp = tgt[:, tt + regen_i : tt + regen_i + self.sentence_length] * 1
        # tgt[:, tt:] = 0
        # tgt[:, tt + regen_i : tt + regen_i + self.sentence_length] = tmp
        orig_tgt = tgt * 1
        for i in range(self.n_sentences_in_paragraph):
            for j in range(self.sentence_length):
                my_mask = self.pseudo_causal_masks[regen_i][
                    : t + 1,
                    : t + 1,
                ]
                decoder_out = self.transformer_decoder(
                    src=tgt,
                    mask=my_mask,
                    is_causal=False,
                )
                decoder_out = self.ln(decoder_out)
                next_logits = self.deproj(decoder_out[:, tt - 1])
                next_logits /= temperature
                v, _ = torch.topk(next_logits, min(topk, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -torch.inf
                # Stochastic
                next_tok = torch.multinomial(next_logits.softmax(-1), 1)
                words.append(next_tok)
                next_emb = self.emb(next_tok)
                # tgt = torch.cat([tgt, next_emb + self.pe[t]], dim=1)
                # Want to skip tokens corresponding to regen_i, but remember we have a batch generation situation...
                tgt[:, [tt]] = next_emb + self.pe[tt - 1]
                # tgt[:, skip_me] = orig_tgt[:, skip_me]
                # t += 1
                tt += 1

        words = torch.cat(words, dim=1)
        articles = words.reshape(
            words.shape[0],
            xyzexpm.shape[1],
            # n_paragraphs,
            self.n_sentences_in_paragraph,
            self.sentence_length,
        )

        # articles[:, -1, regen_i] = paragraphs[:, -1, regen_i]

        x_hat = self.convert_articles_to_xyzexpms(articles)
        return x_hat


def ddpm_schedules(beta1, beta2, T):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta = betas_for_alpha_bar(
        T,
        lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
    )

    # beta = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha = 1 - beta
    alpha_bar = torch.cumsum(torch.log(alpha), dim=0).exp()
    alpha_bar_prev = torch.cat(
        [torch.tensor([1.0], device=alpha_bar.device), alpha_bar[:-1]]
    )
    posterior_variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
    sigma = torch.sqrt(beta)

    return {
        "alpha": alpha,  # \alpha_t
        "alpha_bar": alpha_bar,  # \bar{\alpha_t}
        "alpha_bar_prev": alpha_bar_prev,
        "posterior_variance": posterior_variance,  # \sigma_{t-1}^2
        "beta": beta,  # \beta (will be used as \sigma_t^2)
        "sigma": sigma,  # \sigma
    }


class MLP(nn.Module):

    def __init__(
        self,
        nail_shape: torch.Size,
        cond_shape: torch.Size,
        hidden_size: int,
        hammer_shape: torch.Size,
    ):
        super().__init__()
        self.nail_shape = nail_shape
        self.cond_shape = cond_shape
        self.hammer_shape = hammer_shape

        self.nail_size = np.prod(nail_shape).item()
        self.cond_size = np.prod(cond_shape).item()
        self.hammer_size = np.prod(hammer_shape).item()

        self.rms_nail = RunningMeanStd(shape=(self.nail_shape[-1],))
        self.rms_cond = RunningMeanStd(shape=(self.cond_shape[-1],))

        self.t_emb = GaussianFourierFeatures(1, hidden_size, 1e1)

        self.layers = nn.ModuleList(
            [
                nn.Linear(self.cond_size + self.nail_size + 1, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(
                    self.cond_size + self.nail_size + 1 + hidden_size, hidden_size
                ),
                nn.LeakyReLU(),
                nn.Linear(
                    self.cond_size + self.nail_size + 1 + hidden_size, hidden_size
                ),
                nn.LeakyReLU(),
                nn.Linear(
                    self.cond_size + self.nail_size + hidden_size + 1, self.hammer_size
                ),
            ]
        )

    def setup(self, x, w):
        self.rms_nail.to(device=x.device)
        self.rms_nail.update(x)

        self.rms_cond.to(device=w.device)
        self.rms_cond.update(w)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        t: torch.Tensor,
        context_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Assume the following shapes
        # `x` is the noisy input, shape (B, T_sequence, H_nail, W_nail)
        # 'w' is the conditioning, shape (B, T_condition, H_cond, W_cond)
        # `t` is the denoising timestep, shape (B,)
        # Both `x` and `w` first get projected to the same Euclidean space and then concatenated
        assert len(x.shape) == 4
        assert len(w.shape) == 4
        assert x.shape[0] == w.shape[0], "Batch size mismatch"
        assert x.shape[1] == self.nail_shape[0]
        assert x.shape[2] == self.nail_shape[1]
        assert x.shape[3] == self.nail_shape[2]
        assert w.shape[1] == self.cond_shape[0]
        assert w.shape[2] == self.cond_shape[1]
        assert w.shape[3] == self.cond_shape[2]

        x = self.rms_nail.normalize(x)
        x = x.reshape(x.shape[0], self.nail_size)

        w = self.rms_cond.normalize(w)
        w = w.reshape(w.shape[0], self.cond_size)
        w[w.isnan()] = 0  # unconditional => only use bias
        if context_mask is not None:
            w[context_mask] = 0

        # temb = self.t_emb(t[:, None])
        temb = t[:, None]

        z = torch.cat([x, w, temb], dim=-1)
        for i, op in enumerate(self.layers):
            if i == 0 or isinstance(op, nn.LeakyReLU):
                z = op.forward(z)
            else:
                z = op.forward(torch.cat([x, w, temb, z], dim=-1))
        y = z.reshape(z.shape[0], *self.hammer_shape)
        return y


class ClassifierFreeDDPM(nn.Module):
    def __init__(self, eps_model, betas, T, device, drop_prob=0.1):
        super().__init__()
        self.eps_model = eps_model

        for k, v in ddpm_schedules(betas[0], betas[1], T).items():
            # populates alpha, alpha_bar, beta, sigma
            self.register_buffer(k, v)

        self.T = T
        self.mse_loss = nn.MSELoss()
        self.device = device
        self.drop_prob = drop_prob

    def setup(self, x, y):
        self.eps_model.setup(x, y)

    def forward(self, x, y):
        # y *= torch.nan

        t = torch.randint(1, self.T, (x.shape[0],)).to(self.device)
        eps = torch.randn_like(x)
        x_t = (
            torch.sqrt(self.alpha_bar[t, None, None, None]) * x
            + torch.sqrt(1 - self.alpha_bar[t, None, None, None]) * eps
        )
        eps_tar = x
        context_mask = torch.bernoulli(
            torch.zeros(y.shape[0], device=y.device) + self.drop_prob
        ).to(dtype=torch.bool, device=self.device)
        eps_out = self.eps_model(x_t, y, t / self.T, context_mask)
        return self.mse_loss(eps_tar, eps_out)

    def generate(self, y, guide_w=0.0):
        # y *= torch.nan

        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance
        n_sample = y.shape[0]
        x_t = torch.randn(n_sample, *self.eps_model.hammer_shape).to(self.device)

        # double the batch
        y_double = y.tile((2, 1, 1, 1))
        context_mask = torch.zeros(2 * n_sample).to(
            device=self.device, dtype=torch.bool
        )
        context_mask[n_sample:] = True  # makes second half of batch context free

        # pbar = tqdm(total=self.T)
        # for t in reversed(range(self.T)):
        for t in reversed(range(self.T)):
            # t = 0
            z = (
                torch.randn(n_sample, *self.eps_model.hammer_shape).to(self.device)
                if t > 1
                else 0
            )
            t_is_double = (
                torch.ones(2 * n_sample, device=self.device, dtype=torch.float)
                * t
                / self.T
            )

            # double batch
            x_t_double = x_t.tile((2, 1, 1, 1))

            x0_pred = self.eps_model(x_t_double, y_double, t_is_double, context_mask)
            x0_pred_1 = x0_pred[:n_sample]
            x0_pred_2 = x0_pred[n_sample:]
            x0_pred = guide_w * x0_pred_1 + (1 - guide_w) * x0_pred_2
            # eps *= 0

            # x_t = (
            #     1
            #     / torch.sqrt(self.alpha[t])
            #     * (x_t - eps * (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t]))
            #     + self.sigma[t] * z
            # )
            # print(x_t)
            # x_t[:, : y.shape[1]] = y
            x_t = x0_pred + torch.sqrt(self.posterior_variance[t]) * z
            # x_t = x0_pred
            # break

            # pbar.update(1)
        # pbar.close()
        return None, x_t


class ConditionalVAE(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self, nail_size: int, hidden_size: int, latent_size: int, cond_size: int
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
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + cond_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(latent_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(latent_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            # nn.Linear(latent_size + hidden_size, hidden_size),
            # nn.LeakyReLU(),
            nn.Linear(latent_size + hidden_size, nail_size),
        )
        # Add mu and log_var layers for reparameterization
        self.mu = nn.Linear(hidden_size, latent_size)
        self.log_var = nn.Linear(hidden_size, latent_size)
        self.hammer_size = latent_size
        self.cond_size = cond_size
        self.nail_size = nail_size

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, w: torch.Tensor):
        x = self.nail_rms.normalize(x * 1)
        x = x.reshape(x.shape[0], -1)
        w = self.nail_rms.normalize(w * 1)
        w[w.isnan()] = 0
        w = w.reshape(w.shape[0], -1)
        encoder_out = self.encoder.forward(torch.cat((x, w), dim=-1))
        encoder_out = encoder_out.reshape(encoder_out.shape[0], -1)
        return encoder_out

    def decode(self, z: torch.Tensor, w: torch.Tensor):
        z = z.reshape(z.shape[0], -1)
        decoder_out = z * 1
        w = self.nail_rms.normalize(w * 1)
        w[w.isnan()] = 0
        w = w.reshape(w.shape[0], -1)
        for j, op in enumerate(self.decoder):
            if isinstance(op, torch.nn.Linear):
                if j > 0:
                    decoder_out_w = torch.cat((z, decoder_out), dim=-1)
                else:
                    decoder_out_w = torch.cat((decoder_out, w), dim=-1)
                decoder_out = op(decoder_out_w)
            else:
                decoder_out = op(decoder_out)
        decoder_out = decoder_out.reshape(decoder_out.shape[0], -1, self.nail_size)
        return decoder_out

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        encoded = self.encode(x, w)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        log_var = torch.clamp(log_var, -5, 5)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z, w)
        return z, decoded, mu, log_var


class ConditionalWAE(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self, nail_size: int, hidden_size: int, latent_size: int, cond_size: int
    ):
        super().__init__()
        self.nail_rms = RunningMeanStd(shape=(nail_size,))
        self.cond_rms = RunningMeanStd(shape=(cond_size,))

        self.encoder = nn.Sequential(
            nn.Linear(nail_size + cond_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(latent_size + cond_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(latent_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(latent_size + hidden_size, nail_size),
        )
        self.discriminator = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(latent_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(latent_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(latent_size + hidden_size, 1),
        )

        self.hammer_size = latent_size
        self.cond_size = cond_size
        self.nail_size = nail_size

    def encode(self, x: torch.Tensor, w: torch.Tensor):
        x = self.nail_rms.normalize(x * 1)
        x = x.reshape(x.shape[0], -1)
        w = self.nail_rms.normalize(w * 1)
        w[w.isnan()] = 0
        w = w.reshape(w.shape[0], -1)
        encoder_out = self.encoder.forward(torch.cat((x, w), dim=-1))
        encoder_out = encoder_out.reshape(encoder_out.shape[0], -1)
        return encoder_out

    def decode(self, z: torch.Tensor, w: torch.Tensor):
        z = z.reshape(z.shape[0], -1)
        decoder_out = z * 1
        w = self.nail_rms.normalize(w * 1)
        w[w.isnan()] = 0
        w = w.reshape(w.shape[0], -1)
        for j, op in enumerate(self.decoder):
            if isinstance(op, torch.nn.Linear):
                if j > 1:
                    decoder_out_w = torch.cat((z, decoder_out), dim=-1)
                else:
                    decoder_out_w = torch.cat((decoder_out, w), dim=-1)
                decoder_out = op(decoder_out_w)
            else:
                decoder_out = op(decoder_out)
        decoder_out = decoder_out.reshape(decoder_out.shape[0], -1, self.nail_size)
        return decoder_out

    def discriminate(self, z: torch.Tensor):
        disc_out = z * 1
        for j, op in enumerate(self.discriminator):
            if isinstance(op, torch.nn.Linear):
                if j > 1:
                    disc_out = torch.cat((z, disc_out), dim=-1)
            disc_out = op(disc_out)
        return disc_out

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        z = self.encode(x, w)
        disc_logits = self.discriminate(z)
        decoded = self.decode(z, w)
        return z, decoded, disc_logits


class ConditionalVAEWithPrior(ConditionalVAE):
    def __init__(
        self, nail_size: int, hidden_size: int, latent_size: int, cond_size: int
    ):
        super().__init__(nail_size, hidden_size, latent_size, cond_size)
        self.prior_encoder = nn.Sequential(
            nn.Linear(cond_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
        )
        self.prior_mu = nn.Linear(hidden_size, latent_size)
        self.prior_log_var = nn.Linear(hidden_size, latent_size)

    def prior_encode(self, w: torch.Tensor):
        w = self.nail_rms.normalize(w * 1)
        w[w.isnan()] = 0
        w = w.reshape(w.shape[0], -1)
        encoder_out = self.prior_encoder.forward(w)
        encoder_out = encoder_out.reshape(encoder_out.shape[0], -1)
        return encoder_out

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        encoded = self.encode(x, w)
        prior_encoded = self.prior_encode(w)

        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        log_var = torch.clamp(log_var, -5, 5)

        prior_mu = self.prior_mu(prior_encoded)
        prior_log_var = self.prior_log_var(prior_encoded)
        prior_log_var = torch.clamp(prior_log_var, -5, 5)

        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z, w)
        return z, decoded, mu, log_var, prior_mu, prior_log_var