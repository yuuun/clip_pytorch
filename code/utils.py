import torch
import torch.nn.functional as F
from typing import Tuple
import torch.nn as nn

# loss function from https://github.com/moein-shariatnia/OpenAI-CLIP/blob/2b512ce83d21f4cbc535d2569b308473fa0a201c/CLIP.py
def calculate_loss(image_embed, text_embed, temperature):
    logits = (text_embed @ image_embed.T)
    image_similarity = image_embed @ image_embed.T
    text_similarity = text_embed @ text_embed.T

    targets = F.softmax((image_similarity + text_similarity) / 2 * temperature, dim=-1)
    text_loss = cross_entropy(logits, targets, reduction='none')
    image_loss = cross_entropy(logits.T, targets.T, reduction='none')

    loss = (image_loss + text_loss) / 2.0
    return loss.mean()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

def metrics(similarity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc