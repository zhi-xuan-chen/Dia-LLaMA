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
from monai.networks.nets.swin_unetr import SwinTransformer

CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]


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

        # self.vision_encoder = SwinTransformer(
        #     in_chans = 1,
        #     embed_dim = 48,
        #     window_size = (7, 7, 7),
        #     patch_size = (2, 2, 2),
        #     depths = (2, 2, 2, 2),
        #     num_heads = (3, 6, 12, 24),
        #     mlp_ratio = 4,
        #     qkv_bias = True,
        #     drop_rate = 0.0,
        #     attn_drop_rate = 0.0,
        #     drop_path_rate = 0.0,
        #     use_checkpoint = False,
        #     spatial_dims = 3,
        # )

        # load pretrained vision encoder from RadFM
        vit3d_ckpt = torch.load(
            '/data/chenzhixuan/checkpoints/LLM4CTRG/RadFM_vit3d.pth', map_location='cpu')
        self.vision_encoder.load_state_dict(vit3d_ckpt, strict=True)
        # vit3d_ckpt = torch.load(
        #     '/data/chenzhixuan/checkpoints/LLM4CTRG/t3d_swin3d.pth', map_location='cpu')
        # self.vision_encoder.load_state_dict(vit3d_ckpt, strict=True)
        # print('load pretrained vision encoder from T3D')

        # frozen the vision encoder
        # for param in self.vision_encoder.parameters():
        #     param.requires_grad = False

        self.vis_dim = vis_dim

        self.perceiver = PerceiverResampler(
            dim=self.vis_dim, num_latents=perceiver_num)
        self.fc = nn.Linear(self.vis_dim, self.embedding_dim)
        # load pretrained perceiver and fc from RadFM
        state_dict = torch.load(
            '/data/chenzhixuan/checkpoints/LLM4CTRG/RadFM_perceiver_fc.pth', map_location='cpu')
        self.perceiver.load_state_dict(state_dict['perceiver'])
        # self.fc.load_state_dict(state_dict['fc'])
        
        # used to classify the disease category
        self.prototype = nn.Parameter(torch.randn((2, 14, 49)), requires_grad=True) # s increase to 2 
        # self.cls_head = nn.Linear(self.vis_dim, 14*4)
        self.group_cls_head = nn.ModuleList([
            nn.Linear(self.vis_dim, 2) for _ in range(14)
        ])
        # remember the pos and neg samples for each category
        self.proj = nn.Linear(self.vis_dim, 768)
        self.pos_sample_memory = nn.Parameter(torch.randn(14, 1, 768))
        self.neg_sample_memory = nn.Parameter(torch.randn(14, 1, 768))
        self.temp = nn.Parameter(0.07 * torch.ones([]))

    def forward(self, vision_x, cls_labels=None, text_input=None, key_words_query=None, mode=None):
        if self.flag == 'Text':
            # 获取输入张量vision_x的形状，并将各个维度的大小赋值给相应的变量
            # B: 批量大小，一次处理的样本数量
            # S: 序列长度，例如在处理文本时，这可能是句子的长度
            # C: 通道数，例如在处理图像时，这可能是颜色通道的数量（红、绿、蓝）
            # H: 高度，例如在处理图像时，这可能是图像的高度
            # W: 宽度，例如在处理图像时，这可能是图像的宽度
            # D: 深度，例如在处理三维数据（如3D图像或点云）时，这可能是数据的深度
            B, S, C, H, W, D = vision_x.shape
            vision_x = rearrange(vision_x, "b S c h w d-> (b S) c h w d")

            vision_x, pos_embedding = self.vision_encoder(vision_x) # b*s, 49, 768
            vision_x = rearrange(
                vision_x, "(b s) v d -> b s v d", b=B, s=S)  # b s v d

            prototype = self.prototype.unsqueeze(0).unsqueeze(-1) # 1, 2, 14, 49, 1
            # softmax prototype
            prototype = F.softmax(prototype, dim=-2)

            group_vision_x = vision_x.unsqueeze(2) # b s 1 v d
            group_vision_x = group_vision_x * prototype # b, s, 14, 49, 768
            group_vision_x = group_vision_x.permute(0, 2, 1, 3, 4).contiguous() # b, 14, s, 49, 768
            group_vision_x = group_vision_x.mean(dim=2) # b, 14, 49, 768
            cls_logits = []
            for i in range(14):
                cls_logits.append(self.group_cls_head[i](group_vision_x[:,i].sum(dim=1)))
            cls_logits = torch.cat(cls_logits, dim=1)

            cls_logits = cls_logits.view(-1, 14, 2)
            cls_logits = cls_logits.permute(0, 2, 1).contiguous()

            # if mode == 'cls':
            #     return cls_logits

            ######### pos-neg contrastive learning #########
            if mode == 'train':
                group_vision_x = group_vision_x.view(-1, 49, 768) # b*14, 49, 768
                group_vision_x = group_vision_x.sum(dim=1) # b*14, 768
                all_memory_samples = []
                for i in range(B):
                    cls_label = cls_labels[i]
                    for n, j in enumerate(cls_label):
                        memory_samples = []
                        current_sample = group_vision_x[i*14+j] # 768
                        if j == 0:
                            # target_sample = (F.softmax(torch.matmul(current_sample, self.neg_sample_memory[n].t()), dim=0).unsqueeze(-1) * self.neg_sample_memory[n]).sum(dim=0)
                            # target_sample = self.neg_sample_memory[n].mean(dim=0)

                            memory_samples.append(self.neg_sample_memory[n])
                            memory_samples.append(self.pos_sample_memory[n])
                            all_memory_samples.append(torch.cat(memory_samples, dim=0))
                        else:
                            # target_sample = (F.softmax(torch.matmul(current_sample, self.pos_sample_memory[n].t()), dim=0).unsqueeze(-1) * self.pos_sample_memory[n]).sum(dim=0)
                            # target_sample = self.pos_sample_memory[n].mean(dim=0)
         
                            memory_samples.append(self.pos_sample_memory[n])
                            memory_samples.append(self.neg_sample_memory[n])
                            all_memory_samples.append(torch.cat(memory_samples, dim=0))
                all_memory_samples = torch.stack(all_memory_samples, dim=0) # b*14, 2, 768
                all_memory_samples_feats = F.normalize(all_memory_samples, dim=-1)
                group_vision_x = self.proj(group_vision_x)
                group_vision_x = group_vision_x.unsqueeze(1) # b*14, 1, 768
                current_sample_feats = F.normalize(group_vision_x, dim=-1)

                sim = torch.matmul(current_sample_feats, all_memory_samples_feats.transpose(-1, -2)).squeeze(1) # b*14, 2

                sim = sim /self.temp
            elif mode == 'cls':
                group_vision_x = group_vision_x.sum(dim=1) # b, 14, 768
                all_sim = []
                for i in range(14):
                    current_sample = group_vision_x[:,i] # b, 768
                    memory_samples = []
                    memory_samples.append(self.pos_sample_memory[i])
                    memory_samples.append(self.neg_sample_memory[i])
                    memory_samples = torch.cat(memory_samples, dim=0) # 20, 768
                    memory_samples = memory_samples.unsqueeze(0).repeat(B, 1, 1) # b, 20, 768
                    memory_samples_feats = F.normalize(memory_samples, dim=-1)
                    current_sample_feats = F.normalize(self.proj(current_sample).unsqueeze(1), dim=-1)
                    sim = torch.matmul(current_sample_feats, memory_samples_feats.transpose(-1, -2)).squeeze(1) # b, 20
                    all_sim.append(sim)
                all_sim = torch.stack(all_sim, dim=1) # b, 14, 20
                return cls_logits, all_sim

            #########################################################

            vision_x = vision_x.unsqueeze(2)  # b s F=1 v d

            loss_matching = None

            # reshapes to (b, S, n, d), n is number of latents
            vision_x = self.perceiver(vision_x)
            # vision_x = checkpoint(self.perceiver,vision_x)

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
        if mode=='train':
            return out_put, cls_logits, loss_matching, sim
        else: 
            return out_put
