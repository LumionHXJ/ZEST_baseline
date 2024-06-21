import os
import torch
import torchaudio
import logging
import numpy as np
import random
from tqdm import tqdm
import random
from speechbrain.pretrained import EncoderClassifier
from speaker_classifier import SpeakerModel

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

classifier = EncoderClassifier.from_hparams(source="/home/huxingjian/model/huggingface/speechbrain/spkrec-ecapa-voxceleb", 
                                            run_opts={"device":"cuda"})
model = torch.load('EASE.pth', map_location=device)
model.to(device)
model.eval()

speaker_dict = {}
for ind in range(11, 21):
    speaker_dict["00"+str(ind)] = ind-11

def getspkrlabel(file_name, front=True):
    if front:
        spkr_name = file_name[0:4]
    else:
        spkr_name = file_name[11:15]
    spkr_label = speaker_dict[spkr_name]
    return spkr_label

def get_speaker_out(loader, ground_truth):
    results = np.array([], dtype=np.int64)
    for i, data in enumerate(tqdm(loader)):
        speaker_feat, labels, emo_labels = data[0].to(device), data[1].to(device), data[2].to(device)
        if ground_truth:
            results = np.concatenate((results, labels.detach().cpu().numpy()))
        else:
            outputs, _, _ = model(speaker_feat)
            results = np.concatenate((results, np.argmax(outputs.detach().cpu().numpy(), axis=1)))
        # break
    return results

def get_speaker_out_test(folder, bs=72):
    preds, labels = np.array([], dtype=np.int64), []
    inputs, filenames = [], os.listdir(folder)
    for i, wav_file in enumerate(tqdm(filenames)):
        sig, sr = torchaudio.load(os.path.join(folder, wav_file))
        embeddings = classifier.encode_batch(sig.cuda())[0, 0, :]
        inputs.append(embeddings)
        if (i+1) % bs == 0 or i == len(filenames)-1:
            speaker_feat = torch.stack(inputs)
            with torch.no_grad():
                outputs, _, _ = model(speaker_feat)
                preds = np.concatenate((preds, np.argmax(outputs.cpu().numpy(), axis=1)))
            inputs = []
        # labels.append(getspkrlabel(wav_file))
        labels.append(getspkrlabel(wav_file, front=False))

    return preds, np.array(labels[:len(preds)], dtype=np.int64)


preds, labels = get_speaker_out_test('DSDT_cxt/')

acc = np.sum(preds==labels) / len(labels)
print('acc:', acc)
