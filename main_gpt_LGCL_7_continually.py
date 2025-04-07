
# ------------------------------------------
# Modified Code from (https://github.com/shreydan/VisionGPT2)
# ------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timm import create_model, list_models
from types import SimpleNamespace
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
import gc
import json

import warnings
import logging
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datetime import datetime
import sys
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from pycocoevalcap.cider.cider import Cider
import random

from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# nltk.download('punkt')
# nltk.download('wordnet')

def extract_nouns(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    nouns = [word for word, pos in tagged if pos.startswith('NN')]
    return nouns

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token

tokenizer.encode_plus('hello! this is a caption')

class Dataset:
    def __init__(self, df, tfms, task_name,use_lgcl):
        self.df = df
        self.tfms = tfms
        self.task_name = task_name
        self.use_lgcl = use_lgcl

    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        sample = self.df.iloc[idx,:]
        image_path = sample['image']
        image = sample['image']
        caption = sample['caption']
        # category = sample['category_name']

        # if self.use_lgcl == True:
        #     task_prompt = f"An image of "
        #     caption = f"{task_prompt}{caption}"

        image = Image.open(image).convert('RGB')
        image = np.array(image)
        augs = self.tfms(image=image)
        image = augs['image']
        caption = f"{caption}<|endoftext|>"
        input_ids = tokenizer(
            caption,
            truncation=True)['input_ids']
        labels = input_ids.copy()
        labels[:-1] = input_ids[1:]
        return image,input_ids,labels,image_path

class Dataset_COCO:
    def __init__(self, df, tfms, task_name,use_lgcl):
        self.df = df
        self.tfms = tfms
        self.task_name = task_name
        self.use_lgcl = use_lgcl

    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        sample = self.df.iloc[idx,:]
        image = sample['image']
        caption = sample['caption']
        # category = sample['category_name']

        image = Image.open(image).convert('RGB')
        image = np.array(image)
        augs = self.tfms(image=image)
        image = augs['image']
        caption = f"{caption}<|endoftext|>"
        input_ids = tokenizer(
            caption,
            truncation=True)['input_ids']
        labels = input_ids.copy()
        labels[:-1] = input_ids[1:]
        return image,input_ids,labels


class VGDataset(Dataset):
    def __init__(self, df, tfms, use_lgcl=False):
        self.df = df
        self.tfms = tfms
        self.use_lgcl = use_lgcl

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        image_path = sample['image']
        caption = sample['caption']

        try:  # Handle potential PIL.UnidentifiedImageError
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

        image = np.array(image)
        augs = self.tfms(image=image)
        image = augs['image']

        caption = f"{caption}<|endoftext|>"
        input_ids = tokenizer(caption, truncation=True)['input_ids']
        labels = input_ids.copy()
        labels[:-1] = input_ids[1:]

        return image, input_ids, labels
    
def collate_fn(batch):
    image = [i[0] for i in batch]
    input_ids = [i[1] for i in batch]
    labels = [i[2] for i in batch]

    image_paths = [i[3] for i in batch]  # Extract image_paths

    image = torch.stack(image,dim=0)
    input_ids = tokenizer.pad(
        {'input_ids':input_ids},
        padding='longest',
        return_attention_mask=False,
        return_tensors='pt'
    )['input_ids']
    labels = tokenizer.pad(
        {'input_ids':labels},
        padding='longest',
        return_attention_mask=False,
        return_tensors='pt'
    )['input_ids']
    mask = (input_ids!=tokenizer.pad_token_id).long()
    labels[mask==0]=-100
    return image, input_ids, labels, image_paths


class GPT2Attention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len
        
        self.c_attn = nn.Linear(self.embed_dim, self.head_size * self.n_heads * 3,bias=True)
        self.scale = self.head_size ** -0.5
        
        self.register_buffer('mask',torch.tril(torch.ones(1,1,self.seq_len,self.seq_len)))
        
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        
        
    def forward(self, x):
        b,t,c = x.shape
        # q,k,v shape individually: batch_size x seq_len x embed_dim
        # we know that qk_t = q x k_t, where q=bxtxhead_dim, k_t=bxhead_timxt
        q,k,v = self.c_attn(x).chunk(3,dim=-1)
        q = q.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3) # batch x n_heads x seq_len x head_dim
        k = k.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3)
        v = v.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3)
        
        qk_t = (q@k.transpose(-2,-1)) * self.scale
        qk_t = qk_t.masked_fill(self.mask[:,:,:t,:t]==0,float('-inf'))
        qk_t = F.softmax(qk_t,dim=-1)
        weights = self.attn_dropout(qk_t)
        
        attention = weights @ v # batch x n_heads x t x head_size
        attention = attention.permute(0,2,1,3).contiguous().view(b,t,c) # batch x t x embed_dim
        
        out = self.c_proj(attention)
        out = self.resid_dropout(out)
        
        return out
    
class GPT2CrossAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len
        
        self.q = nn.Linear(self.embed_dim,self.embed_dim)
        self.k = nn.Linear(self.embed_dim,self.embed_dim)
        self.v = nn.Linear(self.embed_dim,self.embed_dim)
        self.scale = self.head_size ** -0.5
        
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        
    def forward(self, q,k,v):
        b,t,c = q.shape
        
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        
        q = q.view(b,q.size(1),self.n_heads,self.head_size).permute(0,2,1,3) # batch x n_heads x seq_len x head_dim
        k = k.view(b,k.size(1),self.n_heads,self.head_size).permute(0,2,1,3)
        v = v.view(b,v.size(1),self.n_heads,self.head_size).permute(0,2,1,3)
        
        qk_t = (q@k.transpose(-2,-1)) * self.scale
        qk_t = F.softmax(qk_t,dim=-1)
        weights = self.attn_dropout(qk_t)
        
        attention = weights @ v # batch x n_heads x t x head_size
        attention = attention.permute(0,2,1,3).contiguous().view(b,t,c) # batch x t x embed_dim
        
        out = self.c_proj(attention)
        out = self.resid_dropout(out)
        
        return out

class GPT2MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.mlp_ratio = config.mlp_ratio
        self.mlp_dropout = config.mlp_dropout
        
        self.c_fc = nn.Linear(self.embed_dim,self.embed_dim*self.mlp_ratio)
        self.c_proj = nn.Linear(self.embed_dim*self.mlp_ratio,self.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.mlp_dropout)
        
    def forward(self,x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class GPT2Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        self.mlp = GPT2MLP(config)
        self.ln_3 = nn.LayerNorm(self.embed_dim)
        self.cross_attn = GPT2CrossAttention(config)
        
    def forward(self,x,enc_out):
        x = x+self.attn(self.ln_1(x))
        x = x+self.cross_attn(self.ln_2(x),enc_out,enc_out)
        x = x+self.mlp(self.ln_3(x))
        return x
    
class VisionGPT2Model(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.config = config
        
        vit = create_model('vit_base_patch16_224',pretrained=True,num_classes=0)
        self.patch_embed = vit.patch_embed
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = vit.cls_token
        embed_len = num_patches + vit.num_prefix_tokens
        self.pos_embed = vit.pos_embed
        self.pos_drop = nn.Dropout(p=0.)
        
        self.blocks = nn.ModuleList([vit.blocks[i] for i in range(config.depth)])
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.embed_dim),
            wpe = nn.Embedding(config.seq_len,config.embed_dim),
            drop = nn.Dropout(config.emb_dropout),
            h = nn.ModuleList([GPT2Block(config) for _ in range(config.depth)]),
            ln_f = nn.LayerNorm(config.embed_dim)
        ))
        self.lm_head = nn.Linear(config.embed_dim,config.vocab_size,bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
    def _pos_embed(self,x):
        pos_embed = self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + pos_embed
        return self.pos_drop(x)
    
    def pretrained_layers_trainable(self,trainable=False):
        layers = [
            self.cls_token, self.patch_embed, self.pos_embed, self.blocks,
            self.transformer.wte, self.transformer.wpe,
            self.transformer.ln_f, self.lm_head
        ]
        gpt_layers = [[
            self.transformer.h[i].ln_1,self.transformer.h[i].ln_2,
            self.transformer.h[i].attn,self.transformer.h[i].mlp
        ] for i in range(self.config.depth)]
        for l in gpt_layers:
            layers.extend(l)
        
        for layer in layers:
            if not isinstance(layer,nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = trainable
            else:
                layer.requires_grad = trainable
                
        total_frozen_params = sum([p.numel() for p in self.parameters() if not p.requires_grad])
        print(f'{total_frozen_params=}')
        
    def unfreeze_gpt_layers(self,):
        gpt_layers = [[
            self.transformer.h[i].ln_1,self.transformer.h[i].ln_2,
            self.transformer.h[i].attn,self.transformer.h[i].mlp
        ] for i in range(self.config.depth)]
        flatten = []
        for l in gpt_layers:
            flatten.extend(l)
            
        for layer in flatten:
            if not isinstance(layer,nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = True
            else:
                layer.requires_grad = True
        
    @classmethod    
    def from_pretrained(self,config):
        model = VisionGPT2Model(config)
        sd = model.state_dict()
        keys = sd.keys()
        ignore_matches = ['blocks.','cross_attn.','ln_3','cls_token','pos_embed','patch_embed.','.attn.mask']
        vit_keys = [key for key in keys if any(match in key for match in ignore_matches)]
        gpt_keys = [key for key in keys if key not in vit_keys]
        
        gpt2_small = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = gpt2_small.state_dict()
        hf_keys = sd_hf.keys()
        hf_keys = [k for k in hf_keys if not k.endswith('.attn.masked_bias')]
        hf_keys = [k for k in hf_keys if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        for k in hf_keys:
            if any(match in k for match in ignore_matches):
                continue
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
            
        model.load_state_dict(sd)
        
        return model
    
    def forward(self,image,input_ids,labels=None):
        
        image = self.patch_embed(image)
        image = self._pos_embed(image)
        
        token_embeddings = self.transformer.wte(input_ids) # batch x seq_len
        pos_embs = torch.arange(0,input_ids.size(1)).to(input_ids.device)
        positional_embeddings = self.transformer.wpe(pos_embs)
        input_ids = self.transformer.drop(token_embeddings+positional_embeddings)
        
        for i in range(self.config.depth):
            image = self.blocks[i](image)
            input_ids = self.transformer.h[i](input_ids, image)
        
        # Use average pooling to get the final image embedding
        image_embedding = image.mean(dim=1)  # Shape: [batch_size, embedding_dim]

        input_ids = self.transformer.ln_f(input_ids)

        if labels is not None:
            lm_logits = self.lm_head(input_ids)
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))

            return loss, image_embedding, lm_logits
        
        loss = 777
        lm_logits = self.lm_head(input_ids[:,[-1],:])
        return lm_logits, loss
    
    def generate(self,image,sequence,max_tokens=50,temperature=1.0,deterministic=False):
        for _ in range(max_tokens):
            out, loss = self(image,sequence)
            logits = out
            out = out[:,-1,:] / temperature
            probs = F.softmax(out,dim=-1)
            if deterministic:
                next_token = torch.argmax(probs,dim=-1,keepdim=True)
            else:
                next_token = torch.multinomial(probs,num_samples=1)
            sequence = torch.cat([sequence,next_token],dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            
        return sequence.cpu().flatten(), logits, loss
    
class Trainer:
    def __init__(self,model_config,train_config, dls, task_num=None, task_name=None, neg_prompt_pool=None, use_lgcl=None, custom_model_name=None):
        
        self.train_config = train_config
        self.model_config = model_config
        self.device = self.train_config.device

        self.task_num = task_num
        self.task_name = task_name
        self.neg_prompt_pool = neg_prompt_pool
        self.current_task_pool = []
        self.use_lgcl = use_lgcl
        self.custom_model_name = custom_model_name

        self.text_encoder = GPT2LMHeadModel.from_pretrained('gpt2')
        
        self.model = VisionGPT2Model.from_pretrained(model_config).to(self.device)
        self.model.pretrained_layers_trainable(trainable=False)
        
        print(f'trainable parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}')
        
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.scaler = GradScaler()
        
        self.train_dl, self.val_dl = dls
        
        total_steps = len(self.train_dl)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.train_config.lr / 25.)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=self.train_config.lr,
            epochs=self.train_config.epochs,
            steps_per_epoch=total_steps
        )
        
#         self.sched = get_linear_schedule_with_warmup(self.optim,num_warmup_steps=0,num_training_steps=total_steps)
        
        self.metrics = pd.DataFrame()
        self.metrics[['train_loss','train_perplexity','val_loss','val_perplexity']] = None
        
        self.gen_tfms = A.Compose([
            A.Resize(224,224),
            A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],always_apply=True),
            ToTensorV2()
        ])
            
    def save_model(self,):
        
        # Get the current timestamp as a string
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.train_config.model_path.mkdir(exist_ok=True)
        sd = self.model.state_dict()

        if self.custom_model_name:  # Save with custom name if provided
            torch.save(sd, self.train_config.model_path / f'{self.custom_model_name}.pt')
        else:
            torch.save(sd,self.train_config.model_path/f'captioner.pt')
            torch.save(sd,self.train_config.model_path/f'captioner_{self.task_num}.pt')
        # torch.save(sd,self.train_config.model_path/f'captioner_{timestamp}.pt')
    
    def save_model_pretrained(self,):

        self.train_config.model_path.mkdir(exist_ok=True)
        sd = self.model.state_dict()
        torch.save(sd,self.train_config.model_path/f'pretrained_model.pt')

        
    def load_best_model(self,):
        sd = torch.load(self.train_config.model_path/'captioner.pt')
        self.model.load_state_dict(sd)
    
    def load_model(self, task_num):
        sd = torch.load(self.train_config.model_path/f'captioner_{task_num}.pt')
        self.model.load_state_dict(sd)
    
    def load_model_pretrained(self):

        # sd = torch.load(self.train_config.model_path/f'pretrained_visual_genome.pt')
        sd = torch.load(self.train_config.model_path/f'pretrained_VG.pt')
        # sd = torch.load(self.train_config.model_path/f'captioner_9.pt')
        self.model.load_state_dict(sd)
    
    def load_model_task(self, task_num):
        sd = torch.load(self.train_config.model_path/f'captioner_{task_num}.pt')
        self.model.load_state_dict(sd)

    def train_one_epoch(self, epoch, task_name, task_num):
        
        prog = tqdm(self.train_dl,total=len(self.train_dl))

        running_loss = 0
        if task_num > 0 and self.use_lgcl == True:
            
            prompt_neg = random.choice(self.neg_prompt_pool)
            input_ids_neg = self.tokenizer(prompt_neg, return_tensors="pt").input_ids

            with torch.no_grad():
                output_neg = self.text_encoder(input_ids_neg, output_hidden_states=True)

                # Hidden states are in the `hidden_states` field, access the last hidden layer
                hidden_states_neg = output_neg.hidden_states

                # Take the last hidden state (which corresponds to the last layer of the model)
                last_hidden_state_neg = hidden_states_neg[-1]

                # Take the mean over the token dimension (dim=1) to get a single vector representation
                text_embedding_neg = last_hidden_state_neg.mean(dim=1)
                text_embedding_neg = text_embedding_neg.to(self.device)

        for image, input_ids, labels, image_path in prog:
            
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                loss, image_embedding, lm_logits = self.model(image,input_ids,labels)

                loss_lgcls = 0.0
                loss_nouns = 0.0
                if self.use_lgcl == True:
                    
                    for input_id in input_ids:
                        caption = self.tokenizer.decode(input_id, skip_special_tokens=True)
                        # caption_clean = caption.replace("An image of ", "").strip()

                        nouns = extract_nouns(caption)
                        noun_string = ""
                        for noun in nouns:
                            if noun_string == "":
                                if noun[0] in ["a", "e", "i", "o", "u"]:
                                    noun_string = "an " + noun
                                else:
                                    noun_string = "a " + noun
                            else:
                                if noun[0] in ["a", "e", "i", "o", "u"]:
                                    noun_string = noun_string + " and an " + noun
                                else:
                                    noun_string = noun_string + " and a " + noun

                        prompt = f"An image of {noun_string}"
                        
                        # add prompt to the negative pool
                        if prompt not in self.current_task_pool:
                            self.current_task_pool.append(prompt)

                        prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=labels.shape[1], truncation=True).input_ids.to(self.device)
                        batch_size = image.shape[0]  # Get batch size
                        prompt_ids = prompt_ids.expand(batch_size, -1)  # Expand along batch dimension

                        with torch.no_grad():
                            loss_nouns, noun_embedding, _ = self.model(image, prompt_ids, labels)
                            loss_nouns += loss_nouns

                        if task_num > 0:
                            
                            input_id = self.tokenizer(prompt, return_tensors="pt").input_ids

                            with torch.no_grad():
                                output = self.text_encoder(input_id, output_hidden_states=True)
                                
                                # Hidden states are in the `hidden_states` field, access the last hidden layer
                                hidden_states = output.hidden_states
                                
                                # Take the last hidden state (which corresponds to the last layer of the model)
                                last_hidden_state = hidden_states[-1]
                                
                                # Take the mean over the token dimension (dim=1) to get a single vector representation
                                text_embedding = last_hidden_state.mean(dim=1)
                                text_embedding = text_embedding.to(self.device)

                            # Cosine similarity between image embedding (anchor) and correct category embedding (positive) cos_sim = F.cosine_similarity(image_embedding, text_embedding)
                            pos_similarity = F.cosine_similarity(image_embedding, text_embedding, dim=-1)
                            
                            # Cosine similarity between image embedding (anchor) and incorrect category embedding (negative)
                            neg_similarity = F.cosine_similarity(image_embedding, text_embedding_neg, dim=-1)
                            
                            # Triplet loss: To Encourage the image embedding to be closer to the positive and farther from the negative
                            loss_lgcl = 1 - pos_similarity + neg_similarity
                            loss_lgcl = loss_lgcl.mean()
                            loss_lgcls += loss_lgcl
                        else:
                            loss_lgcls = 0

                    loss_lgcls = loss_lgcls / (len(input_ids))
                    loss_nouns = loss_nouns / (len(input_ids))

                # Dynamically adjust the weighting based on the relative magnitudes of the losses:
                # alpha = loss / (loss + loss_lgcls)  # Gives more weight to the dominant loss
                # beta = loss_lgcls / (loss + loss_lgcls)
                # loss = alpha * loss + beta * loss_lgcls + loss_nouns
                # Compute dynamic weights for 3 losses
                total_loss = loss + loss_lgcls + loss_nouns
                alpha = loss / total_loss
                beta = loss_lgcls / total_loss
                gamma = loss_nouns / total_loss

                print(f"loss {loss}")
                print(f"loss_lgcls {loss_lgcls}")
                print(f"loss_nouns {loss_nouns}")
                # loss = alpha * loss + beta * loss_lgcls + gamma * loss_nouns  # 1st test - Dynamic
                loss = loss + loss_lgcls + loss_nouns                         # 2nd test - SUM
                # loss = loss + beta * loss_lgcls + loss_nouns                  # Best results so far
                print(f"combined loss {loss}")

                # Prevents numerical instability when one loss is significantly larger than the other.
                # loss = torch.log(torch.exp(loss) + torch.exp(loss_lgcls))
                
                # Normal loss addition
                # loss = loss + loss_lgcls + loss_nouns
                # loss = loss + loss_lgcls

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)
                
                running_loss += loss.item()

                prog.set_description(f'train loss: {loss.item():.3f}')
                
            del image, input_ids, labels, loss
            
        train_loss = running_loss / len(self.train_dl)
        train_pxp = np.exp(train_loss)
        
        self.metrics.loc[epoch,['train_loss','train_perplexity']] = (train_loss,train_pxp)
        
    @torch.no_grad()
    def valid_one_epoch(self,epoch):
        
        prog = tqdm(self.val_dl,total=len(self.val_dl))
        
        running_loss = 0.
        accuracy_list = []
        average_bleus_list = []
        cider_scores_list = []

        # Initialize the CIDEr scorer
        cider_scorer = Cider()
        references_dict = {}  # Dictionary to hold references (target captions)
        hypotheses_dict = {}  # Dictionary to hold hypotheses (predicted captions)

        
        csv_path = r"coco_captions.csv"  # Replace with actual CSV path
        df = pd.read_csv(csv_path)
        
        for batch_idx, (image, input_ids, labels, image_path) in enumerate(prog):
            
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                loss, image_emb, lm_logits = self.model(image,input_ids,labels)

                # probabilities = torch.softmax(lm_logits, dim=-1)
                # predicted_tokens = torch.argmax(probabilities, dim=-1)

                # print(f"lm_logits {lm_logits}")
                probabilities = F.softmax(lm_logits,dim=-1)
                predicted_tokens = torch.multinomial(probabilities.view(-1, probabilities.size(-1)), num_samples=1).view(probabilities.size(0), probabilities.size(1))  # Sample from the softmax distribution for all tokens in all time steps

                # Use smoothing function to handle lack of higher-order n-gram matches for BLEU Scores
                smoothing_function = SmoothingFunction().method1

                decoded_captions = []
                captions_list = []
                bleu_scores_list = []

                bleu_1_list = []
                bleu_2_list = []
                bleu_3_list = []
                bleu_4_list = []
                rouge1_list = []
                rouge2_list = []
                rougeL_list = []
                meteor_list = []
                for i in range(len(predicted_tokens)):
                    
                    predicted_token = predicted_tokens[i]
                    caption_token = input_ids[i]

                    img_path = image_path[i]  # Get current image path
                    target_captions = df[df["image"] == img_path]["caption"].str.lower().tolist()
                    # print(f"target_captions {target_captions}")

                    predicted_token = predicted_token.tolist()
                    caption_token = caption_token.tolist()

                    predicted_caption = tokenizer.decode(predicted_token, skip_special_tokens=True)
                    target_caption = tokenizer.decode(caption_token, skip_special_tokens=True)
                    
                    decoded_captions.append(predicted_caption)
                    captions_list.append(target_caption)

                    predicted_caption = nltk.word_tokenize(predicted_caption.lower())
                    target_caption = target_caption.split()

                    # print(f"predicted_caption {predicted_caption}")

                    # print(f"target_caption {target_caption}")
                    
                    bleu_score = 0
                    
        ########################
                    # Use smoothing function to avoid 0 scores for short captions

                average_bleu = 0
                average_bleus_list.append(average_bleu)

                correct_predictions = (predicted_tokens == input_ids).sum().item()

                total_tokens = input_ids.numel()
                accuracy = correct_predictions / total_tokens
                accuracy_list.append(accuracy)

                running_loss += loss.item()
                
                prog.set_description(f'valid loss: {loss.item():.3f}')
                
            del image, input_ids, labels, loss

        cider_scores_list.append(0)

        # print(f"cider_scores_list {cider_scores_list}")
            
        val_loss = running_loss / len(self.val_dl)
        val_pxp = np.exp(val_loss)
        
        self.metrics.loc[epoch,['val_loss','val_perplexity']] = (val_loss,val_pxp)
        
        average_accuracy = 0
        average_bleus = 0
        average_cider = 0

        average_bleu_1 = 0
        average_bleu_2 = 0
        average_bleu_3 = 0
        average_bleu_4 = 0
        average_rouge1 = 0
        average_rouge2 = 0
        average_rougeL = 0
        average_meteor = 0

        print(f"average_bleu_1 training {average_bleu_1}")
        print(f"average_bleu_2 training {average_bleu_2}")
        print(f"average_bleu_3 training {average_bleu_3}")
        print(f"average_bleu_4 training {average_bleu_4}")
        print(f"average_rouge1 training {average_rouge1}")
        print(f"average_rouge2 training {average_rouge2}")
        print(f"average_rougeL training {average_rougeL}")
        print(f"average_meteor training {average_meteor}")

        return val_pxp, val_loss, average_accuracy, average_bleus, average_cider, average_bleu_1, average_bleu_2, average_bleu_3, average_bleu_4, average_rouge1, average_rouge2, average_rougeL, average_meteor
        
    def clean(self):
        gc.collect()
        torch.cuda.empty_cache()
       
    
    def fit(self, task_num, task_name):
        
        best_pxp = 1e9
        best_epoch = -1
        prog = tqdm(range(self.train_config.epochs))
        
        training_data_list = []
        for epoch in prog:
            
            if epoch == self.train_config.freeze_epochs_gpt:
                self.model.unfreeze_gpt_layers()
                print('unfreezing GPT2 entirely...')
                
            if epoch == self.train_config.freeze_epochs_all:
                self.model.pretrained_layers_trainable(trainable=True)
            
            self.model.train()
            prog.set_description('training')
            self.train_one_epoch(epoch, task_name, task_num)
            self.clean()
            
            self.model.eval()
            prog.set_description('validating')
            pxp, loss, average_accuracy, average_bleus, average_cider, average_bleu_1, average_bleu_2, average_bleu_3, average_bleu_4, average_rouge1, average_rouge2, average_rougeL, average_meteor = self.valid_one_epoch(epoch)
            self.clean()
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            print(self.metrics.tail(1))
            
            if pxp < best_pxp:
                best_pxp = pxp
                best_epoch = epoch
                print('saving best model...')
                self.save_model()

                try:
                    training_data_list.append(f'{pxp},{loss},{average_accuracy},{average_bleus},{average_cider},{average_bleu_1}, {average_bleu_2}, {average_bleu_3}, {average_bleu_4}, {average_rouge1}, {average_rouge2}, {average_rougeL}, {average_meteor},{task},{timestamp}\n')
                except:
                    task = "VG"
                    training_data_list.append(f'{pxp},{loss},{average_accuracy},{average_bleus},{average_cider},{average_bleu_1}, {average_bleu_2}, {average_bleu_3}, {average_bleu_4}, {average_rouge1}, {average_rouge2}, {average_rougeL}, {average_meteor},{task},{timestamp}\n')

        with open('training_output.csv', 'a') as file:
        
            # Write the values in a comma-separated format
            file.write(training_data_list[-1])

        for i in self.current_task_pool:
            self.neg_prompt_pool.append(i)

        return {
            'best_perplexity': best_pxp,
            'best_epoch': best_epoch
        }
    
    def fit_forgetting(self, task):
        
        best_pxp = 1e9
        best_epoch = -1
        prog = tqdm(range(self.train_config.epochs))
        
        for epoch in range(1):
            
            if epoch == self.train_config.freeze_epochs_gpt:
                self.model.unfreeze_gpt_layers()
                print('unfreezing GPT2 entirely...')
                
            if epoch == self.train_config.freeze_epochs_all:
                self.model.pretrained_layers_trainable(trainable=True)

            self.model.eval()
            prog.set_description('validating')
            pxp, loss, average_accuracy, average_bleus, average_cider, average_bleu_1, average_bleu_2, average_bleu_3, average_bleu_4, average_rouge1, average_rouge2, average_rougeL, average_meteor = self.valid_one_epoch(epoch)
            self.clean()
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open('forgetting_output.csv', 'a') as file:
                
                # Write the values in a comma-separated format
                file.write(f'{pxp},{loss},{average_accuracy},{average_bleus},{average_cider},{average_bleu_1}, {average_bleu_2}, {average_bleu_3}, {average_bleu_4}, {average_rouge1}, {average_rouge2}, {average_rougeL}, {average_meteor},{task},{timestamp}\n')

            print(self.metrics.tail(1))
            
        return {
            'best_perplexity': best_pxp,
            'best_epoch': best_epoch
        }           
        
    @torch.no_grad()
    def generate_caption(self,image,max_tokens=50,temperature=1.0,deterministic=False):
        
        self.model.eval()
        
        image = Image.open(image).convert('RGB')
        image = np.array(image)
        image = self.gen_tfms(image=image)['image']
        image = image.unsqueeze(0).to(self.device)
        sequence = torch.ones(1,1).to(device=self.device).long() * self.tokenizer.bos_token_id
        
        tokens, logits, loss = self.model.generate(
            image,
            sequence,
            max_tokens=max_tokens,
            temperature=temperature,
            deterministic=deterministic
        )
        caption = self.tokenizer.decode(tokens.numpy(),skip_special_tokens=True)
        
        return caption, tokens, logits, loss
    
    def generate_caption_with_prompt(self, image, prompt="This is a picture of", max_tokens=50, temperature=1.0, deterministic=False):
        """
        Generate a caption using a language-based prompt, even if the model wasn't explicitly trained with prompts.
        """
        self.model.eval()
        
        # Load and transform the image
        image = Image.open(image).convert('RGB')
        image = np.array(image)
        image = self.gen_tfms(image=image)['image']
        image = image.unsqueeze(0).to(self.device)
        
        # Tokenize the prompt
        prompt_input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Start the sequence with the prompt
        sequence = torch.cat([prompt_input_ids, torch.ones(1, 1).to(self.device).long() * self.tokenizer.bos_token_id], dim=1)
        
        tokens, logits, loss = self.model.generate(
            image,
            sequence,
            max_tokens=max_tokens,
            temperature=temperature,
            deterministic=deterministic
        )
        
        caption = self.tokenizer.decode(tokens.numpy(), skip_special_tokens=True)
        
        return caption, tokens, logits, loss

if __name__ == '__main__':
    model_config = SimpleNamespace(
        vocab_size = 50_257,
        embed_dim = 768,
        num_heads = 12,
        seq_len = 1024,
        depth = 12,
        attention_dropout = 0.1,
        residual_dropout = 0.1,
        mlp_ratio = 4,
        mlp_dropout = 0.1,
        emb_dropout = 0.1,
    )

    epochs_train = 10
    train_config = SimpleNamespace(
        epochs = epochs_train,
        freeze_epochs_gpt = int(epochs_train/3),
        freeze_epochs_all = int(epochs_train/2),
        lr = 1e-4, # 1e-4
        device = 'cuda',
        model_path = Path('captioner'),
        batch_size = 32,
        shuffle_tasks = True,
        use_lgcl = False,
        used_dataset = "MSCOCO", # "MSCOCO",
        pretrain_VG = False,
        pretrain_MSCOCO = False
    )

    sample_tfms = [
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(),
        A.ColorJitter(),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),
        A.HueSaturationValue(p=0.3),
    ]
    train_tfms = A.Compose([
        *sample_tfms,
        A.Resize(224,224),
        A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],always_apply=True),
        ToTensorV2()
    ])
    valid_tfms = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],always_apply=True),
        ToTensorV2()
    ])

    if train_config.pretrain_VG == True:
        ### Pretrain on Visual Genome
        df = pd.read_csv("Visual Genome/visual_genome_data.csv")  # Load VG dataset
        df = df[['image', 'caption']]

        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        train_ds = VGDataset(train_df, train_tfms, use_lgcl=train_config.use_lgcl)
        val_ds = VGDataset(val_df, valid_tfms, use_lgcl=train_config.use_lgcl)

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True, collate_fn=collate_fn)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=train_config.batch_size, shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True, collate_fn=collate_fn)

        # Initialize Trainer
        trainer = Trainer(model_config, train_config, (train_dl, val_dl), task_num=0, task_name="VG", neg_prompt_pool=[], use_lgcl=train_config.use_lgcl, custom_model_name="pretrained_visual_genome")
        trainer.fit(task_num=0, task_name="VG")

        sys.exit(0)

    if train_config.pretrain_MSCOCO == True:
        ### Pretrain on MS COCO
        df = pd.read_csv("coco_captions.csv")  # Load VG dataset
        df = df[['image', 'caption']]

        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        train_ds = VGDataset(train_df, train_tfms, use_lgcl=train_config.use_lgcl)
        val_ds = VGDataset(val_df, valid_tfms, use_lgcl=train_config.use_lgcl)

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True, collate_fn=collate_fn)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=train_config.batch_size, shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True, collate_fn=collate_fn)

        # Initialize Trainer
        trainer = Trainer(model_config, train_config, (train_dl, val_dl), task_num=0, task_name="VG", neg_prompt_pool=[], use_lgcl=train_config.use_lgcl, custom_model_name="pretrained_visual_genome")
        trainer.load_model_pretrained()

        trainer.fit(task_num=0, task_name="MS COCO")

        sys.exit(0)

    train_config = SimpleNamespace(
        epochs = 5,     # 10
        freeze_epochs_gpt = 2,
        freeze_epochs_all = 3,
        lr = 1e-4,
        device = 'cuda',
        model_path = Path('captioner'),
        batch_size = 32,
        shuffle_tasks = False,
        use_lgcl = True,
        used_dataset = "cont_cap", #  "MSCOCO"
        train_true = True,
        forg_calc = True
    )

    if train_config.used_dataset == "MSCOCO":
        # continual training data MS COCO custom dataset with 10 tasks, 1000 images, 5 captions per image
        df = pd.read_csv("ms_coco_data_1000_5.csv")
        df = df[['image_path', 'caption', 'category_name']]
        df = df.rename(columns={'image_path': 'image'})
        df = df.reset_index(drop=True)

        task_list = df["category_name"].unique().tolist()
    elif train_config.used_dataset == "FLICKR8K":

        image_path = r"data/FLICKR8K/Images/"
        df = pd.read_csv("data/FLICKR8K/Flickr8k.token.txt", sep='\t', header=None, names=['image_caption', 'caption'])
        df['image'] = df['image_caption'].apply(lambda x: x.split('#')[0])
        df = df[['image', 'caption']]

        df['image'] = image_path + df['image']
        
        # Remove rows where the 'image' column contains '.1'
        df = df[~df['image'].str.contains(r'\.1$', regex=True)]

        num_tasks = 10 
        task_names = [i for i in range(num_tasks)]

        df['category_name'] = np.random.choice(task_names, size=len(df))
        task_list = df["category_name"].unique().tolist()
    elif train_config.used_dataset == "cont_cap":
        df = pd.read_csv("cont_cap_dataset.csv")
        df = df[['Image Path', 'Captions', 'Subfolder', 'Folder Name']]
        df = df.rename(columns={'Image Path': 'image'})
        df = df.rename(columns={'Captions': 'caption'})

        subfolders = ["person", "sports ball", "tv", "toilet", "bottle"]
        df = df[df['Subfolder'].isin(subfolders)]
        df = df.reset_index(drop=True)

        task_list = subfolders  # task_list = ["person", "sports ball", "tv", "toilet", "bottle"]

    if train_config.used_dataset != "cont_cap":
        train_df, val_df = train_test_split(df,test_size=0.1, random_state=42)
        train_df.reset_index(drop=True,inplace=True)
        val_df.reset_index(drop=True,inplace=True)
    else:
        train_df  = df[df['Folder Name'] == r'C:\Test\LGCL _cap\dataset\processed\train']
        val_df = df[df['Folder Name'] == r'C:\Test\LGCL _cap\dataset\processed\val']
        test_df = df[df['Folder Name'] == r'C:\Test\LGCL _cap\dataset\processed\test\val']

        train_df.reset_index(drop=True,inplace=True)
        val_df.reset_index(drop=True,inplace=True)
        test_df.reset_index(drop=True,inplace=True)

    print(len(train_df),len(val_df))
    
    if train_config.shuffle_tasks == True:
        random.shuffle(task_list)

        # Randomizer seed
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)

    ### implementation of continual training:
    neg_prompt_pool = []    # Global prompt pool to store all positive prompts

    for i in range(1):
        train_true = train_config.train_true
        task_num = 0
        if train_true:
            for task in task_list:

                print(f"task: {task}")
                if train_config.used_dataset != "cont_cap":
                    train_df_task = train_df[train_df['category_name'] == task]
                    val_df_task = val_df[val_df['category_name'] == task]
                else:
                    train_df_task = train_df[train_df['Subfolder'] == task]
                    val_df_task = val_df[val_df['Subfolder'] == task]
                
                task_name = task
            
                train_ds = Dataset(train_df_task,train_tfms, task_name, use_lgcl=train_config.use_lgcl)
                val_ds = Dataset(val_df_task,valid_tfms, task_name, use_lgcl=train_config.use_lgcl)

                train_dl = torch.utils.data.DataLoader(train_ds,batch_size=train_config.batch_size,shuffle=train_config.shuffle_tasks,pin_memory=True,num_workers=2,persistent_workers=True,collate_fn=collate_fn)
                val_dl = torch.utils.data.DataLoader(val_ds,batch_size=train_config.batch_size,shuffle=train_config.shuffle_tasks,pin_memory=True,num_workers=2,persistent_workers=True,collate_fn=collate_fn)

                trainer = Trainer(model_config,train_config,(train_dl,val_dl), task_num, task_name, neg_prompt_pool, train_config.use_lgcl)
                if task_num == 0:
                    trainer.load_model_pretrained()
                else:
                    trainer.load_model_task(task_num=task_num-1)

                trainer.fit(task_num, task_name)

                trainer.metrics

                # METRICS
                trainer.load_model_task(task_num=task_num)

                if train_config.used_dataset != "cont_cap":
                    csv_path = r"coco_captions.csv"  # Replace with actual CSV path
                    df_captions = pd.read_csv(csv_path)
                else:
                    df_captions = pd.read_csv("cont_cap_dataset.csv")
                    df_captions = df_captions.rename(columns={'Image Path': 'image'})
                    df_captions = df_captions.rename(columns={'Captions': 'caption'})

                det = True

                bleu_1_list = []
                bleu_2_list = []
                bleu_3_list = []
                bleu_4_list = []
                rouge1_list = []
                rouge2_list = []
                rougeL_list = []
                meteor_list = []

                from pycocoevalcap.cider.cider import Cider
                # Initialize CIDEr scorer
                cider_scorer = Cider()
                cider_dict_count = 0
                gen_caption_dict = {}
                actual_caption_dict = {}

                test_df_train = test_df[test_df['Subfolder'] == task]

                for image in test_df_train["image"]:

                    plt.imshow(Image.open(image).convert('RGB'))
                    t = np.random.uniform(0.5,1.5)
                    
                    gen_caption, tokens, logits, loss = trainer.generate_caption(image,temperature=t,deterministic=det)
                    
                    actual_captions = df_captions[df_captions["image"] == image]["caption"].str.lower().tolist()

                    # print(f"Generated Caption: {gen_caption}\nTemperature: {t}\nDeterministic: {det}")
                    # print("Actual Captions:")
                    # for cap in actual_captions:
                    #     print(f"- {cap}")

                    # Tokenize the captions
                    gen_caption_tokens = gen_caption.split()
                    actual_caption_tokens = [cap.split() for cap in actual_captions]  # List of lists

                    # Use smoothing function to avoid 0 scores for short captions
                    smoothing_function = SmoothingFunction().method1

                    # Compute BLEU scores with different n-gram weights
                    bleu_1_score = sentence_bleu(actual_caption_tokens, gen_caption_tokens, 
                                                weights=(1.0, 0.0, 0.0, 0.0),  # BLEU-1 (only unigrams)
                                                smoothing_function=smoothing_function)

                    bleu_2_score = sentence_bleu(actual_caption_tokens, gen_caption_tokens, 
                                                weights=(0.5, 0.5, 0.0, 0.0),  # BLEU-2 (unigrams + bigrams)
                                                smoothing_function=smoothing_function)

                    bleu_3_score = sentence_bleu(actual_caption_tokens, gen_caption_tokens, 
                                                weights=(0.33, 0.33, 0.33, 0.0),  # BLEU-3 (up to trigrams)
                                                smoothing_function=smoothing_function)

                    bleu_4_score = sentence_bleu(actual_caption_tokens, gen_caption_tokens, 
                                                weights=(0.25, 0.25, 0.25, 0.25),  # BLEU-4 (up to 4-grams)
                                                smoothing_function=smoothing_function)

                    # Print BLEU scores
                    # print(f"BLEU-1 Score: {bleu_1_score:.4f}")
                    # print(f"BLEU-2 Score: {bleu_2_score:.4f}")
                    # print(f"BLEU-3 Score: {bleu_3_score:.4f}")
                    # print(f"BLEU-4 Score: {bleu_4_score:.4f}")

                    from rouge_score import rouge_scorer
                    from nltk.translate.meteor_score import meteor_score
                    from nltk.tokenize import word_tokenize

                    # Initialize ROUGE scorer
                    rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

                    # Compute ROUGE scores
                    rouge_scores = rouge_scorer.score(' '.join(gen_caption_tokens), ' '.join(actual_caption_tokens[0]))  # Compare with the first reference

                    # Print ROUGE scores
                    # print(f"ROUGE-1 Score: {rouge_scores['rouge1'].fmeasure:.4f}")
                    # print(f"ROUGE-2 Score: {rouge_scores['rouge2'].fmeasure:.4f}")
                    # print(f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure:.4f}")

                    # Tokenize the generated caption and actual captions
                    gen_caption_tokens = word_tokenize(gen_caption.lower())
                    actual_caption_tokens = [word_tokenize(cap.lower()) for cap in actual_captions]

                    # Compute METEOR score
                    meteor_scores = [meteor_score([ref], gen_caption_tokens) for ref in actual_caption_tokens]
                    meteor_score_avg = sum(meteor_scores) / len(meteor_scores)

                    # Add captions to the CIDEr dictionary
                    gen_caption_dict[cider_dict_count] = [gen_caption]
                    actual_caption_dict[cider_dict_count] = actual_captions
                    cider_dict_count += 1

                    bleu_1_list.append(bleu_1_score)
                    bleu_2_list.append(bleu_2_score)
                    bleu_3_list.append(bleu_3_score)
                    bleu_4_list.append(bleu_4_score)
                    rouge1_list.append(rouge_scores['rouge1'].fmeasure)
                    rouge2_list.append(rouge_scores['rouge2'].fmeasure)
                    rougeL_list.append(rouge_scores['rougeL'].fmeasure)
                    meteor_list.append(meteor_score_avg)

                    # Print METEOR score
                    # print(f"METEOR Score: {meteor_score_avg:.4f}")

                average_bleu_1 = sum(bleu_1_list) / len(bleu_1_list)
                average_bleu_2 = sum(bleu_2_list) / len(bleu_2_list)
                average_bleu_3 = sum(bleu_3_list) / len(bleu_3_list)
                average_bleu_4 = sum(bleu_4_list) / len(bleu_4_list)
                average_rouge1 = sum(rouge1_list) / len(rouge1_list)
                average_rouge2 = sum(rouge2_list) / len(rouge2_list)
                average_rougeL = sum(rougeL_list) / len(rougeL_list)
                average_meteor = sum(meteor_list) / len(meteor_list)

                cider_score, _ = cider_scorer.compute_score(actual_caption_dict, gen_caption_dict)

                print(f"average_bleu_1 {average_bleu_1}")
                print(f"average_bleu_2 {average_bleu_2}")
                print(f"average_bleu_3 {average_bleu_3}")
                print(f"average_bleu_4 {average_bleu_4}")
                print(f"average_rouge1 {average_rouge1}")
                print(f"average_rouge2 {average_rouge2}")
                print(f"average_rougeL {average_rougeL}")
                print(f"average_meteor {average_meteor}")
                print(f"average_cider {cider_score}")

                with open('validation_metrics.csv', 'a') as file:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # Write the values in a comma-separated format
                    file.write((f'{average_bleu_1}, {average_bleu_2}, {average_bleu_3}, {average_bleu_4}, {average_rouge1}, {average_rouge2}, {average_rougeL}, {average_meteor},{cider_score},{task}, use_lgcl: {train_config.use_lgcl}, shuffle_tasks: {train_config.shuffle_tasks},{timestamp}\n'))

                # sys.exit(0)
                task_num += 1
        
        # Forgetting calculation
        task_model = 4
        forg_calc = train_config.forg_calc
        if forg_calc:
            for task in range(1): #  for task in range(len(task_list)-1, -1, -1):
                
                print(f"task: {task_list[task], task}")
                if train_config.used_dataset != "cont_cap":
                    train_df_task = train_df[train_df[train_df['Subfolder'] == task]]
                    val_df_task = val_df[train_df[train_df['Subfolder'] == task]]
                else:
                    train_df_task = train_df
                    val_df_task = val_df

                task_name = task_list[task_model]
            
                train_ds = Dataset(train_df_task,train_tfms, task_name, use_lgcl=train_config.use_lgcl)
                val_ds = Dataset(val_df_task,valid_tfms, task_name, use_lgcl=train_config.use_lgcl)

                train_dl = torch.utils.data.DataLoader(train_ds,batch_size=train_config.batch_size,shuffle=train_config.shuffle_tasks,pin_memory=True,num_workers=2,persistent_workers=True,collate_fn=collate_fn)
                val_dl = torch.utils.data.DataLoader(val_ds,batch_size=train_config.batch_size,shuffle=train_config.shuffle_tasks,pin_memory=True,num_workers=2,persistent_workers=True,collate_fn=collate_fn)

                trainer = Trainer(model_config,train_config,(train_dl,val_dl), task_model, task_name, train_config.use_lgcl)
                trainer.load_model(task_model)

                # trainer.fit_forgetting(task)

                trainer.metrics

                # METRICS
                if train_config.used_dataset != "cont_cap":
                    csv_path = r"coco_captions.csv"  # Replace with actual CSV path
                    df_captions = pd.read_csv(csv_path)
                else:
                    df_captions = pd.read_csv("cont_cap_dataset.csv")
                    df_captions = df_captions[['Image Path', 'Captions']]
                    df_captions = df_captions.rename(columns={'Image Path': 'image'})
                    df_captions = df_captions.rename(columns={'Captions': 'caption'})

                det = True

                bleu_1_list = []
                bleu_2_list = []
                bleu_3_list = []
                bleu_4_list = []
                rouge1_list = []
                rouge2_list = []
                rougeL_list = []
                meteor_list = []
                cider_list = []

                from pycocoevalcap.cider.cider import Cider
                # Initialize CIDEr scorer
                cider_scorer = Cider()
                cider_dict_count = 0
                gen_caption_dict = {}
                actual_caption_dict = {}

                subfolders = ["person", "sports ball", "tv", "toilet", "bottle"]
                test_df_train = test_df[test_df['Subfolder'].isin(subfolders)]

                import json
                from datetime import datetime
                from collections import defaultdict

                # Initialize a dictionary to store all results
                results_dict = {
                    "metadata": {
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "model_config": {
                            "use_lgcl": train_config.use_lgcl,
                            "shuffle_tasks": train_config.shuffle_tasks
                        }
                    },
                    "images": []
                }

                # Initialize CIDEr-related structures
                gen_caption_dict = {}
                actual_caption_dict = {}
                cider_dict_count = 0

                # Initialize metric lists for averages
                metric_lists = defaultdict(list)

                test_df_train = test_df_train.drop_duplicates(subset=['image'], keep='first')
                print(f"Processing {len(test_df_train)} unique images")
                for image in test_df_train["image"]:
                    # Load and display image (optional)
                    plt.imshow(Image.open(image).convert('RGB'))
                    t = np.random.uniform(0.5, 1.5)
                    
                    # Generate caption
                    gen_caption, tokens, logits, loss = trainer.generate_caption(image, temperature=t, deterministic=det)
                    
                    # Get actual captions
                    actual_captions = df_captions[df_captions["image"] == image]["caption"].str.lower().tolist()
                    
                    # Extract image ID (assuming it's in the filename or path)
                    image_id = image.split('/')[-1].split('.')[0]  # Modify this based on your image naming
                    
                    # Tokenize captions
                    gen_caption_tokens = gen_caption.split()
                    actual_caption_tokens = [cap.split() for cap in actual_captions]
                    
                    # Compute BLEU scores
                    smoothing_function = SmoothingFunction().method1
                    bleu_scores = {
                        "bleu_1": sentence_bleu(actual_caption_tokens, gen_caption_tokens, 
                                            weights=(1.0, 0, 0, 0), 
                                            smoothing_function=smoothing_function),
                        "bleu_2": sentence_bleu(actual_caption_tokens, gen_caption_tokens, 
                                            weights=(0.5, 0.5, 0, 0), 
                                            smoothing_function=smoothing_function),
                        "bleu_3": sentence_bleu(actual_caption_tokens, gen_caption_tokens, 
                                            weights=(0.33, 0.33, 0.33, 0), 
                                            smoothing_function=smoothing_function),
                        "bleu_4": sentence_bleu(actual_caption_tokens, gen_caption_tokens, 
                                            weights=(0.25, 0.25, 0.25, 0.25), 
                                            smoothing_function=smoothing_function)
                    }
                    
                    from rouge_score import rouge_scorer
                    from nltk.translate.meteor_score import meteor_score
                    from nltk.tokenize import word_tokenize

                    # Initialize ROUGE scorer
                    rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

                    # Compute ROUGE scores
                    rouge_scores = rouge_scorer.score(' '.join(gen_caption_tokens), ' '.join(actual_caption_tokens[0]))  # Compare with the first reference
                    
                    # Compute METEOR score
                    meteor_scores = [meteor_score([ref], word_tokenize(gen_caption.lower())) 
                                    for ref in [word_tokenize(cap.lower()) for cap in actual_captions]]
                    meteor_score_avg = sum(meteor_scores) / len(meteor_scores)
                    
                    # Store for CIDEr calculation
                    gen_caption_dict[cider_dict_count] = [gen_caption]
                    actual_caption_dict[cider_dict_count] = actual_captions
                    cider_dict_count += 1
                    
                    # Create entry for this image
                    image_entry = {
                        "image_id": image_id,
                        "image_path": image,
                        "generated_caption": gen_caption,
                        "actual_captions": actual_captions,
                        "scores": {
                            "bleu": bleu_scores,
                            "rouge": {
                                "rouge1": rouge_scores['rouge1'].fmeasure,
                                "rouge2": rouge_scores['rouge2'].fmeasure,
                                "rougeL": rouge_scores['rougeL'].fmeasure
                            },
                            "meteor": meteor_score_avg,
                            "temperature": t,
                            "deterministic": det
                        }
                    }
                    
                    results_dict["images"].append(image_entry)
                    
                    # Update metric lists for averages
                    for metric, value in bleu_scores.items():
                        metric_lists[metric].append(value)
                    metric_lists["rouge1"].append(rouge_scores['rouge1'].fmeasure)
                    metric_lists["rouge2"].append(rouge_scores['rouge2'].fmeasure)
                    metric_lists["rougeL"].append(rouge_scores['rougeL'].fmeasure)
                    metric_lists["meteor"].append(meteor_score_avg)

                # Compute CIDEr score
                cider_score, _ = cider_scorer.compute_score(actual_caption_dict, gen_caption_dict)
                results_dict["metadata"]["average_scores"] = {
                    "bleu_1": sum(metric_lists["bleu_1"]) / len(metric_lists["bleu_1"]),
                    "bleu_2": sum(metric_lists["bleu_2"]) / len(metric_lists["bleu_2"]),
                    "bleu_3": sum(metric_lists["bleu_3"]) / len(metric_lists["bleu_3"]),
                    "bleu_4": sum(metric_lists["bleu_4"]) / len(metric_lists["bleu_4"]),
                    "rouge1": sum(metric_lists["rouge1"]) / len(metric_lists["rouge1"]),
                    "rouge2": sum(metric_lists["rouge2"]) / len(metric_lists["rouge2"]),
                    "rougeL": sum(metric_lists["rougeL"]) / len(metric_lists["rougeL"]),
                    "meteor": sum(metric_lists["meteor"]) / len(metric_lists["meteor"]),
                    "cider": cider_score
                }

                # Save to JSON file
                output_filename = f"caption_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_filename, 'w') as f:
                    json.dump(results_dict, f, indent=2)

                print(f"Results saved to {output_filename}")

                # Also save averages to CSV (keeping your original code)
                with open('forgetting_metrics.csv', 'a') as file:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    avg = results_dict["metadata"]["average_scores"]
                    file.write((f'{avg["bleu_1"]}, {avg["bleu_2"]}, {avg["bleu_3"]}, {avg["bleu_4"]}, '
                            f'{avg["rouge1"]}, {avg["rouge2"]}, {avg["rougeL"]}, {avg["meteor"]},'
                            f'{avg["cider"]},{task}, use_lgcl: {train_config.use_lgcl}, '
                            f'shuffle_tasks: {train_config.shuffle_tasks},{timestamp}\n'))
        see_1_caption = True
        if see_1_caption:

            train_df_task = train_df
            val_df_task = val_df

            task_name = task_list[task_model]
            
            train_ds = Dataset(train_df_task,train_tfms, task_name,train_config.use_lgcl)
            val_ds = Dataset(val_df_task,valid_tfms, task_name,train_config.use_lgcl)

            train_dl = torch.utils.data.DataLoader(train_ds,batch_size=train_config.batch_size,shuffle=True,pin_memory=True,num_workers=2,persistent_workers=True,collate_fn=collate_fn)
            val_dl = torch.utils.data.DataLoader(val_ds,batch_size=train_config.batch_size,shuffle=False,pin_memory=True,num_workers=2,persistent_workers=True,collate_fn=collate_fn)

            trainer = Trainer(model_config,train_config,(train_dl,val_dl), task_model, task_name,train_config.use_lgcl)

            # trainer.load_model_pretrained()
            trainer.load_model(4)

            trainer.model.eval()

            det = True

            image = r"C:\Test\LGCL _cap\data\FLICKR8K\archive\Flickr_Data\Flickr_Data\Images\225909073_25c3c33a29.jpg"
            image = r"C:\Test\LGCL _cap\data/MSCOCO/train2014/COCO_train2014_000000342532.jpg" 
            image = r"C:\Test\LGCL _cap\data/MSCOCO/val2014/COCO_val2014_000000305351.jpg"
            image = r"Untitled.jpg"

            plt.imshow(Image.open(image).convert('RGB'))
            t = np.random.uniform(0.5,1.5)
            
            gen_caption, tokens, logits, loss = trainer.generate_caption(image,temperature=t,deterministic=det)

            actual_captions = [
                "A cat on a bed.",
                "An orange and white cat laying on the bed."
            ]

            print(f"Generated Caption: {gen_caption}\nTemperature: {t}\nDeterministic: {det}")
            print("Actual Captions:")
            for cap in actual_captions:
                print(f"- {cap}")

            # Tokenize the captions
            gen_caption_tokens = gen_caption.split()
            actual_caption_tokens = [cap.split() for cap in actual_captions]  # List of lists

            # Use smoothing function to avoid 0 scores for short captions
            smoothing_function = SmoothingFunction().method1

            # Compute BLEU scores with different n-gram weights
            bleu_1_score = sentence_bleu(actual_caption_tokens, gen_caption_tokens, 
                                        weights=(1.0, 0.0, 0.0, 0.0),  # BLEU-1 (only unigrams)
                                        smoothing_function=smoothing_function)

            bleu_2_score = sentence_bleu(actual_caption_tokens, gen_caption_tokens, 
                                        weights=(0.5, 0.5, 0.0, 0.0),  # BLEU-2 (unigrams + bigrams)
                                        smoothing_function=smoothing_function)

            bleu_3_score = sentence_bleu(actual_caption_tokens, gen_caption_tokens, 
                                        weights=(0.33, 0.33, 0.33, 0.0),  # BLEU-3 (up to trigrams)
                                        smoothing_function=smoothing_function)

            bleu_4_score = sentence_bleu(actual_caption_tokens, gen_caption_tokens, 
                                        weights=(0.25, 0.25, 0.25, 0.25),  # BLEU-4 (up to 4-grams)
                                        smoothing_function=smoothing_function)

            # Print BLEU scores
            print(f"BLEU-1 Score: {bleu_1_score:.4f}")
            print(f"BLEU-2 Score: {bleu_2_score:.4f}")
            print(f"BLEU-3 Score: {bleu_3_score:.4f}")
            print(f"BLEU-4 Score: {bleu_4_score:.4f}")

            from rouge_score import rouge_scorer
            from nltk.translate.meteor_score import meteor_score
            from nltk.tokenize import word_tokenize

            # Initialize ROUGE scorer
            rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

            # Compute ROUGE scores
            rouge_scores = rouge_scorer.score(' '.join(gen_caption_tokens), ' '.join(actual_caption_tokens[0]))  # Compare with the first reference

            # Print ROUGE scores
            print(f"ROUGE-1 Score: {rouge_scores['rouge1'].fmeasure:.4f}")
            print(f"ROUGE-2 Score: {rouge_scores['rouge2'].fmeasure:.4f}")
            print(f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure:.4f}")

            # Tokenize the generated caption and actual captions
            gen_caption_tokens = word_tokenize(gen_caption.lower())
            actual_caption_tokens = [word_tokenize(cap.lower()) for cap in actual_captions]

            # Compute METEOR score
            meteor_scores = [meteor_score([ref], gen_caption_tokens) for ref in actual_caption_tokens]
            meteor_score_avg = sum(meteor_scores) / len(meteor_scores)

            # Print METEOR score
            print(f"METEOR Score: {meteor_score_avg:.4f}")

            