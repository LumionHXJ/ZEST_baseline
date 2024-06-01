# %%
import json
import re
import os.path as osp
import os
import shutil
from tqdm import tqdm

# %%
def getemolabel(file_no):
    file_name = int(file_no)
    if file_name <=350:
        return 'Neutral'
    elif file_name > 350 and file_name <=700:
        return 'Angry'
    elif file_name > 700 and file_name <= 1050:
        return 'Happy'
    elif file_name > 1050 and file_name <= 1400:
        return 'Sad'
    else:
        return 'Surprise'

# %%
file_templete = 'code/{}_esd.txt'
mode = {'train': [], 'val': [], 'test': []}
data_prefix = '/data/huxingjian/Emotion Speech Dataset' # raw datas of ESD
wav_pattern = r"(\d+)_(\d+)\.wav"
data_midfix = 'Mandarin'

assert data_midfix in ['English', 'Mandarin']

# build dir
if not osp.exists(osp.join(data_prefix, data_midfix)):
    os.makedirs(osp.join(data_prefix, data_midfix))
for m in mode.keys():
    if not osp.exists(osp.join(data_prefix, data_midfix, m)):
        os.makedirs(osp.join(data_prefix, data_midfix, m))

# make dataset
for m in mode.keys():
    with open(file_templete.format(m)) as f:
        for l in tqdm(f.readlines()):
            d = json.loads((l.replace("'", '"').strip()))
            wav_file: str = d['audio'].split('/')[-1] # example: 0016_000651.wav
            matched = re.match(wav_pattern, wav_file)
            speaker = matched.group(1)
            file_no = matched.group(2)
            emotion = getemolabel(file_no)
            if data_midfix == 'Mandarin':
                wav_file = wav_file.replace(speaker, str(int(speaker) - 10).zfill(4), 1)
                speaker = str(int(speaker) - 10).zfill(4) # 0016 -> 0006
            mode[m].append(dict(
                src = osp.join(data_prefix, speaker, emotion, wav_file),
                tgt = osp.join(data_prefix, data_midfix, m, wav_file)
            ))
            if not osp.exists(mode[m][-1]['tgt']):
                shutil.copy2(mode[m][-1]['src'], mode[m][-1]['tgt'])
    with open(osp.join(data_prefix, data_midfix, f'{m}.log'), mode='w') as f:
        for l in mode[m]:
            f.writelines(str(l) + '\n')