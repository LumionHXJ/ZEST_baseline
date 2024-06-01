import json
import re
import os.path as osp
import os
import shutil
from tqdm import tqdm


file_templete = 'code/{}_esd.txt'
mode = ('train', 'val', 'test')
data_prefix = '/data/huxingjian/Emotion Speech Dataset' # raw datas of ESD
wav_pattern = r"(\d+)_(\d+)\.wav"
data_midfix = 'English'
assert data_midfix in ['English', 'Mandarin']

# make dataset
for m in mode:
    datas = []
    with open(file_templete.format(m)) as f:
        for l in tqdm(f.readlines()):
            d = json.loads((l.replace("'", '"').strip()))
            wav_file: str = d['audio'].split('/')[-1] # example: 0016_000651.wav
            matched = re.match(wav_pattern, wav_file)
            speaker = matched.group(1)
            file_no = matched.group(2)
            if data_midfix == 'Mandarin':
                wav_file = wav_file.replace(speaker, str(int(speaker) - 10).zfill(4), 1)
                speaker = str(int(speaker) - 10).zfill(4) # 0016 -> 0006
            d['audio'] = osp.join(data_prefix, data_midfix, m, wav_file)
            datas.append(d)
    with open(file_templete.format(m), mode='w') as f:
        for l in datas:
            f.writelines(str(l) + '\n')