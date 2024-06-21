import torch
import joblib
import json
import numpy as np
import torchaudio
import argparse
import librosa
import sys
import os
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from speechbrain.pretrained import EncoderClassifier
from EASE.speaker_classifier import SpeakerModel
from F0_predictor.config import hparams
sys.path.append('code/F0_predictor')
from F0_predictor.pitch_attention_adv import PitchModel
sys.path.append('code/HiFi_GAN')
from HiFi_GAN.models import CodeGenerator, AttrDict
from HiFi_GAN.inference import generate
from scipy.io.wavfile import write


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)
        
def get_hubert_quantization(audio):
    km = ApplyKmeans('code/km.bin')
    processor = Wav2Vec2FeatureExtractor.from_pretrained("/home/huxingjian/model/huggingface/facebook/hubert-base-ls960")
    model = HubertModel.from_pretrained("/home/huxingjian/model/huggingface/facebook/hubert-base-ls960").to("cuda")

    sig, sr = torchaudio.load(audio)
    input_values = processor(sig[0], return_tensors="pt", sampling_rate=16000).input_values  # Batch size 1
    outputs = model(input_values.to("cuda")).last_hidden_state[0]
    return km(outputs)[None]

def get_ease_features(audio):
    classifier = EncoderClassifier.from_hparams(source="/home/huxingjian/model/huggingface/speechbrain/spkrec-ecapa-voxceleb", 
                                                run_opts={"device":"cuda"})
    sig, sr = torchaudio.load(audio)
    embeddings = classifier.encode_batch(sig.cuda())[0, 0, :]
    model = torch.load('EASE.pth', map_location='cuda')
    model.to('cuda')
    model.eval()
    _, _, embedded = model(embeddings)
    return embedded[None]

def get_f0_features(audio, ease, hubert):
    # receive target speech
    model = PitchModel(hparams).to('cuda')
    model.load_state_dict(torch.load('f0_predictor.pth', map_location='cuda'))
    model.eval()
    sig = torchaudio.load(audio)[0]
    mask = torch.tensor([hubert.shape[-1]]).cuda()
    pitch_pred, _, _, _ = model(sig.cuda(), torch.tensor(hubert).cuda(), ease, mask)
    pitch_pred = torch.exp(pitch_pred) - 1
    return pitch_pred

def get_sace_features(audio):
    model = PitchModel(hparams).to('cuda')
    model.load_state_dict(torch.load('f0_predictor.pth', map_location='cuda'))
    model.eval()
    sig = torchaudio.load(audio)[0].numpy()
    inputs = model.processor(sig, sampling_rate=16000, return_tensors="pt")
    _, _, emo_hidden, _ = model.encoder(inputs['input_values'][None].to('cuda'), 1.0)
    return emo_hidden

def construct_results(ease, sace, hubert, f0):
    with open('code/HiFi_GAN/hubert_alladv.json') as f:
        json_config = json.load(f)
    h = AttrDict(json_config)
    generator = CodeGenerator(h).to('cuda')

    state_dict_g = torch.load("cp_hifigan/g_00090000")
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    code = dict(
        code = torch.tensor(hubert).to('cuda'),
        f0 = f0.detach().unsqueeze(0).to('cuda'),
        spkr = ease.detach(),
        emo_embed =sace.detach()
    )
    audio, _ = generate(h, generator, code)
    return audio

def main():
    parser = argparse.ArgumentParser(description="Process source, target, and result speech files.")
    parser.add_argument('--source', type=str, required=True, help='Path to the source speech file')
    parser.add_argument('--target', type=str, required=True, help='Path to the target speech file')
    parser.add_argument('--result', type=str, required=True, help='Path to save the result speech file')
    
    args = parser.parse_args()
    hubert = get_hubert_quantization(args.source)
    ease = get_ease_features(args.source)
    sace = get_sace_features(args.target)
    f0 = get_f0_features(args.target, ease, hubert)
    print(f0)
    audio = construct_results(ease, sace, hubert, f0)
    audio = librosa.util.normalize(audio.astype(np.float32))
    write(args.result, 16000, audio)

if __name__ == '__main__':
    main()