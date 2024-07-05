import torch.nn as nn
import torch.nn.functional as F
import torch
from .helpers import PerceiverResampler
from .utils import get_visual_encoder
from einops import rearrange, repeat
from einops_exts import rearrange_many
import torchvision
from .vit_3d import ViT
from einops.layers.torch import Rearrange
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
import random
from transformers import AutoTokenizer, AutoModel


class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings=32000, embedding_dim=4096, perceiver_num=32, vis_dim=768, patch_size=32, frame_patch_size=4, seg_channel=256):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.torch.randn((num_embeddings, embedding_dim)), requires_grad=True)  # NOTE: will be initialized using the weight from MedLLaMA
        self.figure_token_weight = nn.Parameter(
            torch.randn((2, embedding_dim)), requires_grad=True)
        self.flag = 'Text'
        self.patch_size = patch_size
        self.frame_patch_size = frame_patch_size
        self.seg_channel = seg_channel

        self.vision_encoder = ViT(
            image_size=512,          # image size
            frames=512,               # max number of frames
            image_patch_size=patch_size,     # image patch size
            frame_patch_size=frame_patch_size,      # frame patch size
            dim=vis_dim,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        # load pretrained vision encoder from RadFM
        # vit3d_ckpt = torch.load(
        #     '/data/chenzhixuan/checkpoints/LLM4CTRG/RadFM_vit3d.pth', map_location='cpu')
        # self.vision_encoder.load_state_dict(vit3d_ckpt, strict=True)

        # frozen the vision encoder
        # for param in self.vision_encoder.parameters():
        #     param.requires_grad = False

        self.vis_dim = vis_dim

        self.perceiver = PerceiverResampler(
            dim=self.vis_dim, num_latents=perceiver_num)
        self.fc = nn.Linear(self.vis_dim, self.embedding_dim)
        # load pretrained perceiver and fc from RadFM
        # state_dict = torch.load(
        #     '/data/chenzhixuan/checkpoints/LLM4CTRG/RadFM_perceiver_fc.pth', map_location='cpu')
        # self.perceiver.load_state_dict(state_dict['perceiver'])
        # self.fc.load_state_dict(state_dict['fc'])

        # used to classify the disease category
        self.prototype = nn.Parameter(
            torch.randn((14, 1024)), requires_grad=True)
        self.group_cls_head = nn.ModuleList([
            nn.Linear(self.vis_dim, 2) for _ in range(14)
        ])
        # remember the pos and neg samples for each category
        self.proj = nn.Linear(self.vis_dim, 768)
        self.pos_sample_memory = nn.Parameter(torch.randn(14, 1, 768))
        self.neg_sample_memory = nn.Parameter(torch.randn(14, 1, 768))
        self.temp = nn.Parameter(0.07 * torch.ones([]))

    def forward(self, vision_x, cls_labels=None, text_input=None, key_words_query=None, mode=None):
        B, S, C, H, W, D = vision_x.shape
        vision_x = rearrange(vision_x, "b S c h w d-> (b S) c h w d")

        vision_x, pos_embedding = self.vision_encoder(vision_x)  # b*s, v, d

        prototype = self.prototype.unsqueeze(0).unsqueeze(-1)  # 1, 14, v, 1
        # softmax prototype
        prototype = F.softmax(prototype, dim=-2)

        group_vision_x = vision_x.unsqueeze(1)  # b*s, 1, v, d
        group_vision_x = group_vision_x * prototype  # b*s, 14, v, d

        cls_logits = []
        for i in range(14):
            cls_logits.append(self.group_cls_head[i](
                group_vision_x[:, i].sum(dim=1)))
        cls_logits = torch.cat(cls_logits, dim=1)

        cls_logits = cls_logits.view(-1, 14, 2)
        cls_logits = cls_logits.permute(0, 2, 1).contiguous()

        ######### pos-neg contrastive learning #########
        if mode == 'train':
            group_vision_x = group_vision_x.view(-1, 1024, 768)  # b*14, v, d
            group_vision_x = group_vision_x.sum(dim=1)
            all_memory_samples = []
            for i in range(B):
                cls_label = cls_labels[i]
                for n, j in enumerate(cls_label):
                    memory_samples = []
                    current_sample = group_vision_x[i*14+j]  # 768
                    if j == 0:
                        memory_samples.append(self.neg_sample_memory[n])
                        memory_samples.append(self.pos_sample_memory[n])
                        all_memory_samples.append(
                            torch.cat(memory_samples, dim=0))
                    else:
                        memory_samples.append(self.pos_sample_memory[n])
                        memory_samples.append(self.neg_sample_memory[n])
                        all_memory_samples.append(
                            torch.cat(memory_samples, dim=0))
            all_memory_samples = torch.stack(
                all_memory_samples, dim=0)  # b*14, 2, 768
            all_memory_samples_feats = F.normalize(all_memory_samples, dim=-1)
            group_vision_x = self.proj(group_vision_x)
            group_vision_x = group_vision_x.unsqueeze(1)  # b*14, 1, 768
            current_sample_feats = F.normalize(group_vision_x, dim=-1)

            sim = torch.matmul(
                current_sample_feats, all_memory_samples_feats.transpose(-1, -2)).squeeze(1)  # b*14, 2

            sim = sim / self.temp
        elif mode == 'cls':
            group_vision_x = group_vision_x.sum(dim=1)  # b, 14, 768
            all_sim = []
            for i in range(14):
                current_sample = group_vision_x[:, i]  # b, 768
                memory_samples = []
                memory_samples.append(self.pos_sample_memory[i])
                memory_samples.append(self.neg_sample_memory[i])
                memory_samples = torch.cat(memory_samples, dim=0)  # 20, 768
                memory_samples = memory_samples.unsqueeze(
                    0).repeat(B, 1, 1)  # b, 20, 768
                memory_samples_feats = F.normalize(memory_samples, dim=-1)
                current_sample_feats = F.normalize(
                    self.proj(current_sample).unsqueeze(1), dim=-1)
                sim = torch.matmul(
                    current_sample_feats, memory_samples_feats.transpose(-1, -2)).squeeze(1)  # b, 20
                all_sim.append(sim)
            all_sim = torch.stack(all_sim, dim=1)  # b, 14, 20
            return all_sim
        #########################################################

        vision_x = rearrange(
            vision_x, "(b s F) v d -> b s F v d", b=B, s=S, F=1)  # b s F v d

        vision_x = self.perceiver(vision_x)

        n = vision_x.shape[2]

        vision_x = rearrange(vision_x, "b s n d -> (b s n) d")
        vision_x = self.fc(vision_x)
        vision_x = rearrange(vision_x, "(b T) d -> b T d", b=B, T=n*S)

        embedding_weight = torch.cat(
            [self.weight, self.figure_token_weight], dim=0)  # num_embeddings+2+4, embedding_dim
        embedding_weight = embedding_weight.unsqueeze(0).repeat(
            B, 1, 1)  # B, num_embeddings+2+4, embedding_dim
        # B, num_embeddings+2+4+n, embedding_dim
        embedding_weight = torch.cat([embedding_weight, vision_x], dim=1)
        text_input = F.one_hot(text_input, embedding_weight.shape[1]).to(
            vision_x.dtype).to(vision_x.device)  # B, N, num_embeddings+2+n
        out_put = torch.matmul(text_input, embedding_weight)
        if mode == 'train':
            return out_put, sim
        else:
            return out_put
