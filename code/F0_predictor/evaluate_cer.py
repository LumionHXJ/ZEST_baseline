import os
import torch
import torchaudio
from einops.layers.torch import Rearrange
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
import logging
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn as nn
import random
from sklearn.metrics import f1_score
from tqdm import tqdm
import random
import torch.nn.functional as F
from config import hparams
import pickle5 as pickle
import ast
from pitch_attention_adv import create_dataset, PitchModel
from torch.autograd import Function
import pandas as pd
from torchmetrics.text import CharErrorRate
import string

torch.set_printoptions(profile="full")
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
torch.autograd.set_detect_anomaly(True)
#CUDA devices enabled
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

processor = Wav2Vec2Processor.from_pretrained("/home/huxingjian/model/huggingface/facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("/home/huxingjian/model/huggingface/facebook/hubert-large-ls960-ft")
model.to(device)
model.eval()

def remove_punctuation(text):
    # 创建一个翻译表，将标点符号映射为 None
    translator = str.maketrans('', '', string.punctuation)
    # 使用 translate 方法去除标点符号
    return text.translate(translator)

def get_text_content(loader, ground_truth):
    results = []
    for i, data in enumerate(tqdm(loader)):
        inputs, names = data["audio"], data["names"]
        if ground_truth:
            for name in names:
                batch = name[:name.find("_")]
                with open(f"/path/to/Emotion Speech Dataset/{batch}/{batch}.txt") as f:
                    splits = f.readlines()[int(name[name.find("_")+1:name.find(".")])-1].split("\t")
                assert splits[0] == name[:name.find(".")]
                results.append(splits[1].upper())
        else:
            input_values = torch.cat([processor(inputs[j], return_tensors="pt", sampling_rate=16000).input_values
                                      for j in range(inputs.shape[0])], dim=0)
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            for j in range(predicted_ids.shape[0]):
                transcription = processor.decode(predicted_ids[j])
                results.append(transcription)
        # break
    return results

def get_text_content_test(folder, bs=72):
    preds, labels = [], []
    inputs, max_length, filenames = [], 0, os.listdir(folder)
    for i, filename in enumerate(tqdm(filenames)):
        inputs.append(torchaudio.load(os.path.join(folder, filename))[0][0, :]) # (len, )
        max_length = max(max_length, inputs[-1].shape[-1])
        if (i+1) % bs == 0 or i == len(filenames)-1:
            for j in range(len(inputs)):
                inputs[j] = torch.concatenate((inputs[j], 
                                            torch.zeros((max_length-inputs[j].shape[0], ))), 
                                            dim=0)
            with torch.no_grad():
                input_values = torch.cat([processor(inputs[j], return_tensors="pt", sampling_rate=16000).input_values
                                          for j in range(len(inputs))], dim=0).to(device)
                logits = model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                for j in range(predicted_ids.shape[0]):
                    transcription = processor.decode(predicted_ids[j])
                    preds.append(transcription)
            inputs, max_length = [], 0
        # labels
        batch = filename[0:4] # ! changing for different functions
        with open(f"/data/huxingjian/Emotion Speech Dataset/{batch}/{batch}.txt") as f:
            splits = f.readlines()[int(filename[5:11])-1].split("\t")
            # splits = f.readlines()[int(filename[16:22])-1].split("\t")
        assert splits[0] == filename[0:11] # ! changing
        labels.append(remove_punctuation(splits[1]).upper())

    return preds, labels[:len(preds)]

preds, labels = get_text_content_test('DSDT_cxt/')

cer = CharErrorRate()(labels, preds).item()
print('cer:', cer)
