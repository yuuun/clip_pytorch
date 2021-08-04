import torch
import torch.nn as nn
from layers import *
import torch.nn.functional as F
from torchvision import models

from transformers import AutoModel, AutoTokenizer, BertTokenizer
from typing import Dict, List, Optional, Tuple
from utils import * 
MAX_LEN = 32

class Tokenizer:
    def __init__(self, tokenizer: BertTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, x: str) -> AutoTokenizer:
        return self.tokenizer(
            x, max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt"
        )

    def decode(self, x: Dict[str, torch.LongTensor]):
        return [self.tokenizer.decode(sentence[:sentence_len]) for sentence, sentence_len in 
                zip(x["input_ids"], target["attention_mask"].sum(axis=-1))]

class VisionEncoder(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        base = models.resnet34(pretrained=True)
        
        d_in = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base
        self.projection = Projection(d_in, d_out)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        projected_vec = self.projection(self.base(x))
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class TextEncoder(nn.Module):
    def __init__(self, text_nhid, d_out: int, text_model) -> None:
        super().__init__()
        self.base = AutoModel.from_pretrained(text_model)
        self.projection = Projection(text_nhid, d_out)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base(**x)[0] # x: dictionary   x.keys()  : input_ids, attention_mask
        out = out[:, 0, :]  # get CLS token output
        projected_vec = self.projection(out)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len

class CLIP(nn.Module):
    def __init__(self, 
                 text_nhid: int = 768,
                 embed_dim: int = 512,
                 text_model = "distilbert-base-multilingual-cased",
                 lr: float = 1e-3,
                 temperature: float = 1.0,
        ) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder(d_out=embed_dim)
        self.text_encoder = TextEncoder(text_nhid, embed_dim, text_model)
        self.tokenizer = Tokenizer(AutoTokenizer.from_pretrained(text_model))
        self.lr = lr
        self.temperature = temperature

    def forward(self, images, text_dev) -> torch.Tensor:
        image_embed = self.vision_encoder(images)
        text_embed = self.text_encoder(text_dev)
        similarity = text_embed @ image_embed.T
        
        loss = calculate_loss(image_embed, text_embed, self.temperature)
        #loss = clip_loss(similarity)
        img_acc, text_acc = metrics(similarity)
        return loss, img_acc, text_acc
    
    def evaluate(self, images, text_dev):
        image_embed = self.vision_encoder(images)
        text_embed = self.text_encoder(text_dev)
        similarity = text_embed @ image_embed.T
        
        val, closest = similarity.topk(5, dim=-1)
        return val, closest