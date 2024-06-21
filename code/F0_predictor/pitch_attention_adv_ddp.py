import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import logging
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import torch.nn as nn
import random
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast, GradScaler
import random
import torch.nn.functional as F
from config import hparams
from config import train_tokens_orig, val_tokens_orig, test_tokens_orig, f0_file
import pickle5 as pickle
import ast
import time
from torch.autograd import Function
from tensorboardX import SummaryWriter
from datetime import datetime

def setup_logger(start_time):
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    writer = SummaryWriter(log_dir=f'run_log/{start_time}')
    return logger, writer

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_checkpoint(checkpoint):
    model_state_dict, optimizer_state_dict, final_val_loss = None, None, None
    start_epoch = 0
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        checkpoint = torch.load(checkpoint, map_location='cpu')
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        final_val_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        start_time = checkpoint['start_time']
    except:
        pass
    return model_state_dict, optimizer_state_dict, final_val_loss, start_epoch, start_time

torch.set_printoptions(profile="full")
log_freq = 10
val_freq = 5
accumulation_grad = 1
SEED = 1234

set_seed(1234)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.enabled = False
torch.cuda.empty_cache()

# init ddp
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

ckpt = None
use_amp = False

class MyDataset(Dataset):

    def __init__(self, folder, token_file):
        self.folder = folder
        wav_files = os.listdir(folder)
        wav_files = [x for x in wav_files if ".wav" in x]
        self.wav_files = wav_files
        self.sr = 16000
        self.tokens = {}
        with open(f0_file, 'rb') as handle:
            self.f0_feat = pickle.load(handle)
        with open(token_file) as f:
            lines = f.readlines()
            for l in lines:
                d = ast.literal_eval(l)
                name, tokens = d["audio"], d["hubert"]
                tokens_l = tokens.split(" ")
                self.tokens[name.split(os.sep)[-1]] = np.array(tokens_l).astype(int)

    def __len__(self):
        return len(self.wav_files) 

    def getemolabel(self, file_name):
        file_name = int(file_name[5:-4])
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

    def getspkrlabel(self, file_name):
        spkr_name = file_name[:4]
        speaker_dict = {}
        for ind in range(11, 21):
            speaker_dict["00"+str(ind)] = ind-11
        speaker_feature = np.load(os.path.join("EASE_embeddings", file_name.replace(".wav", ".npy")))

        return speaker_feature, speaker_dict[spkr_name]
        
    def __getitem__(self, audio_ind): 
        class_id = self.getemolabel(self.wav_files[audio_ind])  
        audio_path = os.path.join(self.folder, self.wav_files[audio_ind])
        (sig, sr) = torchaudio.load(audio_path)
        
        sig = sig.numpy()[0, :]
        tokens = self.tokens[self.wav_files[audio_ind]]
        speaker_feat, speaker_label = self.getspkrlabel(self.wav_files[audio_ind])
        
        final_sig = sig
        f0 = self.f0_feat[self.wav_files[audio_ind]]

        return final_sig, f0, tokens, class_id, speaker_feat, speaker_label, self.wav_files[audio_ind]

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class WAV2VECModel(nn.Module):
    def __init__(self,
                 wav2vec,
                 output_dim,
                 hidden_dim_emo):
        
        super().__init__()
        
        self.wav2vec = wav2vec
        
        embedding_dim = wav2vec.config.to_dict()['hidden_size']
        self.out = nn.Linear(hidden_dim_emo, output_dim)
        self.out_spkr = nn.Linear(hidden_dim_emo, 10)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim_emo, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim_emo, out_channels=hidden_dim_emo, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim_emo, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(in_channels=hidden_dim_emo, out_channels=hidden_dim_emo, kernel_size=5, padding=2)

        self.relu = nn.ReLU()
        
    def forward(self, aud, alpha):
        aud = aud.squeeze(0)
        hidden_all = list(self.wav2vec(aud).hidden_states)
        embedded = sum(hidden_all)
        embedded = embedded.permute(0, 2, 1)

        emo_embedded = self.relu(self.conv1(embedded))
        emo_embedded = self.relu(self.conv2(emo_embedded))
        emo_embedded = emo_embedded.permute(0, 2, 1)
        emo_hidden = torch.mean(emo_embedded, 1).squeeze(1)

        out_emo = self.out(emo_hidden)

        reverse_feature = ReverseLayerF.apply(embedded, alpha)

        embedded_spkr = self.relu(self.conv3(reverse_feature))
        embedded_spkr = self.relu(self.conv4(embedded_spkr))
        hidden_spkr = torch.mean(embedded_spkr, -1).squeeze(-1)
        output_spkr = self.out_spkr(hidden_spkr)
        
        return out_emo, output_spkr, emo_hidden, emo_embedded

class CrossAttentionModel(nn.Module):
    def __init__(self, hidden_dim_q, hidden_dim_k):
        super().__init__()
        HIDDEN_SIZE = 256
        NUM_ATTENTION_HEADS = 4
        self.inter_dim = HIDDEN_SIZE//NUM_ATTENTION_HEADS
        self.num_heads = NUM_ATTENTION_HEADS
        self.fc_q = nn.Linear(hidden_dim_q, self.inter_dim*self.num_heads)
        self.fc_k = nn.Linear(hidden_dim_k, self.inter_dim*self.num_heads)
        self.fc_v = nn.Linear(hidden_dim_k, self.inter_dim*self.num_heads)

        self.multihead_attn = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                    self.num_heads,
                                                    dropout = 0.5,
                                                    bias = True,
                                                    batch_first=True)
                                                                                                           
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(hidden_dim_q, eps = 1e-6)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim_q, eps = 1e-6)
        self.fc = nn.Linear(self.inter_dim*self.num_heads, hidden_dim_q)
        self.fc_1 = nn.Linear(hidden_dim_q, hidden_dim_q)
        self.relu = nn.ReLU()
    
    def forward(self, query_i, key_i, value_i):
        query = self.fc_q(query_i)
        key = self.fc_k(key_i)
        value = self.fc_v(value_i)
        cross, _ = self.multihead_attn(query, key, value, need_weights = True)
        skip = self.fc(cross)
 
        skip += query_i
        skip = self.relu(skip)
        skip = self.layer_norm(skip)

        new = self.fc_1(skip)
        new += skip
        new = self.relu(new)
        out = self.layer_norm_1(new)
        
        return out

class PitchModel(nn.Module):
    def __init__(self, hparams):
        super(PitchModel, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained("/home/huxingjian/model/huggingface/facebook/wav2vec2-large-robust-ft-swbd-300h")
        self.wav2vec = Wav2Vec2ForCTC.from_pretrained("/home/huxingjian/model/huggingface/facebook/wav2vec2-large-robust-ft-swbd-300h", 
                                                      output_hidden_states=True)
        self.encoder = WAV2VECModel(self.wav2vec, 5, hparams["emotion_embedding_dim"])
        self.embedding = nn.Embedding(101, 128, padding_idx=100)        
        self.fusion = CrossAttentionModel(128, 128)
        self.linear_layer = nn.Linear(128, 1)
        self.leaky = nn.LeakyReLU()
        self.cnn_reg1 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)
        self.cnn_reg2 = nn.Conv1d(128, 1, kernel_size=(1,), padding=0)
        self.speaker_linear = nn.Linear(128, 128)

    def forward(self, aud, tokens, speaker, lengths, alpha=1.0):
        hidden = self.embedding(tokens.int()) # hubert
        inputs = self.processor(aud, sampling_rate=16000, return_tensors="pt") # torchaudio signal
        emo_out, spkr_out, _, emo_embedded = self.encoder(inputs['input_values'].to(device), alpha) # SACE
        speaker_temp = speaker.unsqueeze(1).repeat(1, emo_embedded.shape[1], 1) # EASE
        speaker_temp = self.speaker_linear(speaker_temp)
        emo_embedded = emo_embedded + speaker_temp # EASE + SACE
        pred_pitch = self.fusion(hidden, emo_embedded, emo_embedded) # Cross-Attn (HuBERT, EASE + SACE)
        pred_pitch = pred_pitch.permute(0, 2, 1)
        pred_pitch = self.cnn_reg2(self.leaky(self.cnn_reg1(pred_pitch))) # Last 1-D CNN
        pred_pitch = pred_pitch.squeeze(1)
        mask = torch.arange(hidden.shape[1]).expand(hidden.shape[0], hidden.shape[1]).to(device) < lengths.unsqueeze(1)
        pred_pitch = pred_pitch.masked_fill(~mask, 0.0)
        mask = mask.int()

        return pred_pitch, emo_out, spkr_out, mask

def custom_collate(data):
    batch_len = len(data)
    new_data = {"audio":[], "mask":[], "labels":[], "hubert":[], "f0":[], "speaker":[], "speaker_label":[], "names":[]}
    max_len_f0, max_len_hubert, max_len_aud = 0, 0, 0
    for ind in range(len(data)):
        max_len_aud = max(data[ind][0].shape[-1], max_len_aud)
        max_len_f0 = max(data[ind][1].shape[-1], max_len_f0)
        max_len_hubert = max(data[ind][2].shape[-1], max_len_hubert)
    for i in range(len(data)):
        final_sig = np.concatenate((data[i][0], np.zeros((max_len_aud-data[i][0].shape[-1]))), -1)
        f0_feat = np.concatenate((data[i][1], np.zeros((max_len_f0-data[i][1].shape[-1]))), -1)
        mask = data[i][2].shape[-1]
        hubert_feat = np.concatenate((data[i][2], 100*np.ones((max_len_f0-data[i][2].shape[-1]))), -1)
        labels = data[i][3]
        speaker_feat = data[i][4]
        speaker_label = data[i][5]
        names = data[i][6]
        new_data["audio"].append(final_sig)
        new_data["f0"].append(f0_feat)
        new_data["mask"].append(torch.tensor(mask))
        new_data["hubert"].append(hubert_feat)
        new_data["labels"].append(torch.tensor(labels))
        new_data["speaker"].append(speaker_feat)
        new_data["speaker_label"].append(speaker_label)
        new_data["names"].append(names)
    new_data["audio"] = np.array(new_data["audio"])
    new_data["mask"] = np.array(new_data["mask"])
    new_data["hubert"] = np.array(new_data["hubert"])
    new_data["f0"] = np.array(new_data["f0"])
    new_data["labels"] = np.array(new_data["labels"])
    new_data["speaker_label"] = np.array(new_data["speaker_label"])
    new_data["speaker"] = np.array(new_data["speaker"])
    return new_data

def create_dataset_ddp(mode, bs=24):
    if mode == 'train':
        folder = "/data/huxingjian/Emotion Speech Dataset/English/train"
        token_file = train_tokens_orig["ESD"]
    elif mode == 'val':
        folder = "/data/huxingjian/Emotion Speech Dataset/English/val"
        token_file = val_tokens_orig["ESD"]
    else:
        folder = "/data/huxingjian/Emotion Speech Dataset/English/test"
        token_file = test_tokens_orig["ESD"]
    dataset = MyDataset(folder, token_file)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True if mode == 'Train' else False,
                    drop_last=False,
                    collate_fn=custom_collate,
                    sampler=sampler)
    return loader, sampler

def train():
    # prepare dataset
    train_loader, train_sampler = create_dataset_ddp("train")
    val_loader, val_sampler = create_dataset_ddp("val")

    # model and loading states
    model = PitchModel(hparams).to(device)
    model_state_dict, optimizer_state_dict, final_val_loss, start_epoch, start_time = load_checkpoint(ckpt)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)

    if local_rank == 0:
        logger, writer = setup_logger(start_time)
    
    # freezing layer in wav2vec2
    unfreeze = [i for i in range(0, 24)] # tuning all layers
    for name, param in model.named_parameters():
        if 'wav2vec' in name:
            param.requires_grad = False
        for num in unfreeze:
            if str(num) in name and 'conv' not in name:
                param.requires_grad = True
    
    if torch.cuda.device_count() > 1:
        if local_rank == 0:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = DDP(model, 
                    device_ids=[local_rank], 
                    find_unused_parameters=True)
    model.to(device)

    base_lr = 1e-4
    parameters = list(model.parameters()) 
    optimizer = Adam([{'params': parameters, 'lr': base_lr}])
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    if final_val_loss is None:
        final_val_loss = 1e20
    scaler = GradScaler()

    # should start when collapsed
    for e in range(start_epoch, 500):
        model.train()
        val_loss, val_acc = 0.0, 0.0
        pred_tr = []
        gt_tr = []
        pred_val = []
        gt_val = []
        pred_tr_sp = []
        gt_tr_sp = []
        pred_val_sp = []
        gt_val_sp = []
        train_sampler.set_epoch(e)
        epoch_start_time = time.time()

        for i, data in enumerate(train_loader):
            model.train()
            p = float(i + e * len(train_loader)) / 100 / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            inputs, mask ,tokens, f0_trg, labels = torch.tensor(data["audio"]).to(device), \
                                                   torch.tensor(data["mask"]).to(device),\
                                                   torch.tensor(data["hubert"]).to(device),\
                                                   torch.tensor(data["f0"]).to(device),\
                                                   torch.tensor(data["labels"]).to(device)
            speaker_label = torch.tensor(data["speaker_label"]).to(device)
            speaker = torch.tensor(data["speaker"]).to(device)
            bs = inputs.size()[0]

            # forward and backward
            if use_amp:
                with autocast():
                    pitch_pred, emo_out, spkr_out, mask_loss = model(inputs, tokens, speaker, mask, alpha)
                    pitch_pred = torch.exp(pitch_pred) - 1
                    loss1 = (mask_loss * nn.L1Loss(reduction='none')(pitch_pred, f0_trg.float().detach())).sum()
                    loss2 = nn.CrossEntropyLoss(reduction='none')(emo_out, labels).sum()
                    loss3 = nn.CrossEntropyLoss(reduction='none')(spkr_out, speaker_label).sum()
                    loss = (loss1 + 1000*loss2 + 1000*loss3) / accumulation_grad
                    scaler.scale(loss).backward()
            else:
                pitch_pred, emo_out, spkr_out, mask_loss = model(inputs, tokens, speaker, mask, alpha)
                pitch_pred = torch.exp(pitch_pred) - 1
                loss1 = (mask_loss * nn.L1Loss(reduction='none')(pitch_pred, f0_trg.float().detach())).sum()
                loss2 = nn.CrossEntropyLoss(reduction='none')(emo_out, labels).sum()
                loss3 = nn.CrossEntropyLoss(reduction='none')(spkr_out, speaker_label).sum()
                loss = (loss1 + 1000*loss2 + 1000*loss3) / accumulation_grad
                loss.backward()
            
            # update
            if (i + 1) % accumulation_grad == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # logging emotions
            pred = torch.argmax(emo_out, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_tr.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_tr.extend(labels)

            # logging speakers
            pred_sp = torch.argmax(spkr_out, dim = 1)
            pred_sp = pred_sp.detach().cpu().numpy()
            pred_sp = list(pred_sp)
            pred_tr_sp.extend(pred_sp)
            labels = speaker_label.detach().cpu().numpy()
            labels = list(labels)
            gt_tr_sp.extend(labels)

            # logging on main thread
            if (i + 1) % log_freq == 0 and local_rank == 0:
                iteration = e * len(train_loader) + i + 1
                writer.add_scalars('Loss',
                                   {"Loss":loss.item() * accumulation_grad / bs,
                                    "L1 Loss":loss1.item() / bs,
                                    "Emo CE":loss2.item() * 1000 / bs,
                                    "Spkr CE":loss3.item() * 1000 / bs},
                                    global_step=iteration)
                writer.add_scalar('Train/Learning Rate',
                                  scalar_value=optimizer.param_groups[0]['lr'], 
                                  global_step=iteration)
                writer.add_text('Time', 
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                iteration)
                logger.info(f"Epoch {e+1}, Iter {i + 1} / {len(train_loader)}, F0 reconstruction Loss {loss1}, LR {optimizer.param_groups[0]['lr']}")
        
        # update with lasting grad
        if (i + 1) % accumulation_grad != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # logging lasting loss
        if (i + 1) % log_freq != 0 and local_rank == 0:
            logger.info(f"Epoch {e+1}, Iter {i + 1} / {len(train_loader)}, F0 reconstruction Loss {loss1}, LR {optimizer.param_groups[0]['lr']}")
            iteration = e * len(train_loader) + i + 1
            writer.add_scalars('Loss',
                                {"Loss":loss.item() * accumulation_grad / bs,
                                "L1 Loss":loss1.item() / bs,
                                "Emo CE":loss2.item() * 1000 / bs,
                                "Spkr CE":loss3.item() * 1000 / bs},
                                global_step=iteration)
            writer.add_scalar('Train/Learning Rate',
                                scalar_value=optimizer.param_groups[0]['lr'], 
                                global_step=iteration)
            writer.add_text('Time', 
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                            iteration)
        if local_rank == 0:
            logger.info(f'Epoch {e+1}, Elapsed Time {time.time() - epoch_start_time}')

        if (e + 1) % val_freq != 0:
            continue

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, mask ,tokens, f0_trg, labels = torch.tensor(data["audio"]).to(device), \
                                                torch.tensor(data["mask"]).to(device),\
                                                torch.tensor(data["hubert"]).to(device),\
                                                torch.tensor(data["f0"]).to(device),\
                                                torch.tensor(data["labels"]).to(device)
                speaker = torch.tensor(data["speaker"]).to(device)
                speaker_label = torch.tensor(data["speaker_label"]).to(device)

                pitch_pred, emo_out, spkr_out, mask_loss = model(inputs, tokens, speaker, mask)
                pitch_pred = torch.exp(pitch_pred) - 1
                
                loss1 = (mask_loss * nn.L1Loss(reduction='none')(pitch_pred, f0_trg.float().detach())).sum()
                loss2 = nn.CrossEntropyLoss(reduction='none')(emo_out, labels).sum()
                loss3 = nn.CrossEntropyLoss(reduction='none')(spkr_out, speaker_label).sum()
                
                loss = loss1 + 1000*loss2 + 1000*loss3

                val_loss += loss1.detach().item()
                pred = torch.argmax(emo_out, dim = 1)
                pred = pred.detach().cpu().numpy()
                pred = list(pred)
                pred_val.extend(pred)
                labels = labels.detach().cpu().numpy()
                labels = list(labels)
                gt_val.extend(labels)

                pred_sp = torch.argmax(spkr_out, dim = 1)
                pred_sp = pred_sp.detach().cpu().numpy()
                pred_sp = list(pred_sp)
                pred_val_sp.extend(pred_sp)

                labels = speaker_label.detach().cpu().numpy()
                labels = list(labels)
                gt_val_sp.extend(labels)

        # saving checkpoints
        if val_loss < final_val_loss and local_rank == 0:
            torch.save({
                'epoch': e+1,  # 保存当前的 epoch
                'model_state_dict': model.module.state_dict(),  # 保存模型状态字典
                'optimizer_state_dict': optimizer.state_dict(),  # 保存优化器状态字典
                'loss': val_loss,  # 保存当前的 loss
                "start_time": start_time,
            }, f'f0_predictor_epoch_{e+1}.pth')
            final_val_loss = val_loss
        
        train_f1 = f1_score(gt_tr, pred_tr, average='weighted')
        train_f1_sp = f1_score(gt_tr_sp, pred_tr_sp, average='weighted')
        val_loss_log = val_loss/len(val_loader)
        val_f1 = f1_score(gt_val, pred_val, average='weighted')
        val_f1_sp = f1_score(gt_val_sp, pred_val_sp, average='weighted')

        # scheduler.step(val_loss_log)
        if local_rank == 0:
            logger.info(f"Epoch {e+1}, Training Accuracy {train_f1} Speaker Acc {train_f1_sp}")
            logger.info(f"Epoch {e+1}, Validation Loss {val_loss_log}, Validation Accuracy {val_f1} Speaker Acc {val_f1_sp}")
            writer.add_scalars("Acc",
                               {'train': train_f1,
                                'val': val_f1},
                                e+1)
            writer.add_scalars('Loss', {"Val L1 Loss": val_loss_log}, e * len(train_loader) + i + 1)
            writer.add_text('Time', 
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                            iteration)
    
    dist.destroy_process_group()
    if local_rank == 0:
        writer.close()


if __name__ == "__main__":
    train()