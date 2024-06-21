import os
import torch
import torchaudio
from einops.layers.torch import Rearrange
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
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

model = PitchModel(hparams).to(device)
model.load_state_dict(torch.load('f0_predictor.pth', map_location=device))
model.eval()

def getemolabel(file_name):
    file_name = int(file_name[-10:-4]) # emotion from target file
    # file_name = int(file_name[5:11])
    if file_name <=350:
        return 0
    elif file_name > 350 and file_name <=700:
        return 1
    elif file_name > 700 and file_name <= 1050:
        return 2
    elif file_name > 1050 and file_name <= 1400:
        return 3
    else:
        return 4

def get_emo_out(loader, ground_truth):
    results = np.array([], dtype=np.int64)
    for i, data in enumerate(tqdm(loader)):
        inputs, labels = torch.tensor(data["audio"]), \
                         torch.tensor(data["labels"])

        if ground_truth:
            results = np.concatenate((results, labels.numpy()))
        else:
            aud, alpha = inputs, 1.0
            with torch.no_grad():
                inputs = model.processor(aud, sampling_rate=16000, return_tensors="pt")
                emo_out, _, _, _ = model.encoder(inputs['input_values'], alpha)
                results = np.concatenate((results, np.argmax(emo_out.numpy(), axis=1)))
        break
    return results

def get_emo_out_test(folder, bs=72):
    preds, labels = np.array([], dtype=np.int64), []
    inputs, max_length, filenames = [], 0, os.listdir(folder)
    for i, filename in enumerate(tqdm(filenames)):
        # preds
        inputs.append(torchaudio.load(os.path.join(folder, filename))[0][0, :])
        max_length = max(max_length, inputs[-1].shape[0])
        if (i+1) % bs == 0 or i == len(filenames)-1:
            for j in range(len(inputs)):
                inputs[j] = torch.concatenate((inputs[j], 
                                            torch.zeros((max_length-inputs[j].shape[0], ))), 
                                            dim=0)
            aud, alpha = torch.stack(inputs).to(device=device, dtype=torch.float32), 1.0
            with torch.no_grad():
                emo_out, _, _, _ = model.encoder(aud, alpha)
                preds = np.concatenate((preds, np.argmax(emo_out.cpu().numpy(), axis=1)))
            inputs, max_length = [], 0
        # labels
        labels.append(getemolabel(filename))

    return preds, np.array(labels[:len(preds)], dtype=np.int64)

preds, labels = get_emo_out_test('DSDT_cxt/')

acc = np.sum(preds==labels) / len(labels)
print('acc:', acc)
