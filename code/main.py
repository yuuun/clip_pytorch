import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import math
import random
from torch.nn import MultiLabelSoftMarginLoss
from sklearn.metrics import classification_report
from Data import Image_Data
import torch.optim as optim
from models import *
from transformers import AutoModel, AutoTokenizer, BertTokenizer

import torch
import time
import pickle


BATCH_SIZE = 64
if __name__=='__main__':

    img_d = Image_Data(BATCH_SIZE)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    model = CLIP()
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(0, 1001):
        total_loss = 0.0 
        total_img_acc = 0.0
        total_cap_acc = 0.0
    
        for i, (img, text) in enumerate(img_d.train_dl):
            start = time.time()
            model.train()
            optimizer.zero_grad()

            img = img.cuda()
            text_dev = {k: v.to(device) for k, v in model.tokenizer(text).items()}
            
            loss, img_acc, cap_acc = model(img, text_dev)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_img_acc += img_acc
            total_cap_acc += cap_acc
            if i % 100 == 0:
                print('Train epoc: {:d} | idx: {:d}| loss: {:.4f} img_acc: {:.4f} cap_acc: {:.4f}  | time: {:.2}s'.format(epoch, i, total_loss/(i+1), total_img_acc/(i+1), total_cap_acc/(i+1), time.time() - start))
        
        with torch.no_grad():
            val_img_acc = 0.0
            val_cap_acc = 0.0
            model.eval()
            for i, (img, text) in enumerate(img_d.valid_dl):
                
                start = time.time() 
                img = img.cuda()
                text_dev = {k: v.to(device) for k, v in model.tokenizer(text).items()}
                
                _, img_acc, cap_acc = model(img, text_dev)
                val_img_acc += img_acc
                val_cap_acc += cap_acc
                if i % 100 == 0:
                    print('Validation epoc: {:d} | idx: {:d}| img_acc: {:.4f} cap_acc: {:.4f}  | time: {:.2}s'.format(epoch, i, val_img_acc/(i+1), val_cap_acc/(i+1), time.time() - start))
                