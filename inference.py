# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json, os
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

# from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
sim_cse_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

# HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
# HIFIGAN_CHECKPT = './checkpts/hifigan.pt'

HIFIGAN_CONFIG = './checkpts/hifigan_universal/config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan_universal/g_02500000'
EMO_DICT_PATH = './resources/filelists/esd/emo_emb.npy'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')
    parser.add_argument('-l', '--length_scale', type=float, required=False, default=1, help='length scale')
    parser.add_argument('-s', '--speaker_ref_path', type=str, default='/home/xingxing/COMP5214/dataset/esd/0020/Angry/evaluation/0020_000354.wav', help='speaker ref path for multispeaker model')
    parser.add_argument('-e', '--emotion', type=str, required=False, default='Neutral', help='emotion text')
    parser.add_argument('-i', '--emo_intensity', type=float, required=False, default='1', help='emotion intensity')
    parser.add_argument('-fd', '--folder', type=str, required=False, default='sample', help='output folder')
    args = parser.parse_args()
    
    # if not isinstance(args.speaker_id, type(None)):
    #     assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
    #     spk = torch.LongTensor([args.speaker_id]).cuda()
    # else:
    #     spk = None
    
    print('Initializing Grad-TTS...')
    generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc)['model_state_dict'])
    _ = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')
    
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    
    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')
    
    spk_emb_dictionary = np.load(params.spk_emb_path, allow_pickle=True).item()
    emo_emb_dictionary = np.load(EMO_DICT_PATH, allow_pickle=True).item()
    
    spk_path = args.speaker_ref_path
    
    # emo = 'Neutral'
    
    fpath = Path(spk_path)
    wav = preprocess_wav(fpath)

    voice_encoder = VoiceEncoder()
    speaker_embed = voice_encoder.embed_utterance(wav)
    
    inputs = tokenizer(args.emotion, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        emo_embs = sim_cse_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    
    spk = torch.from_numpy(speaker_embed).cuda()
    emo = emo_embs[0].cuda()
    neutral = torch.from_numpy(emo_emb_dictionary['Neutral']).cuda()

    output_folder = f'./out/{args.folder}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    f = open(f'{output_folder}/info.txt', "a")
    f.write(f"timestamps: {args.timesteps}\nlength_scale: {args.length_scale}\nspeaker_ref_path: {args.speaker_ref_path}\nemotion: {args.emotion}")
    f.close()

    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'Synthesizing {i} text...', end=' ')
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
            
            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5, stoc=False, spk=spk, emo=emo, length_scale=1, e_neutral=neutral, e_mix=args.emo_intensity)
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')

            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                        
            write(f'{output_folder}/sample_{i}_{args.timesteps}.wav', 22050, audio)

    print('Done. Check out `out` folder for samples.')
