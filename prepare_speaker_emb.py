import os, glob, tqdm
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

base_dir = "/home/xingxing/COMP5214/dataset/esd"
spk_emb_dict = {}

encoder = VoiceEncoder()

def get_speak_emb(spk):
    spk_file_list = glob.glob(os.path.join(base_dir, spk)+'/*/*/*.wav')
    print(spk_file_list)
    wavs = [preprocess_wav(spf) for spf in spk_file_list]
    embed = encoder.embed_speaker(wavs)
    spk_emb_dict[spk] = embed
    print(spk)
    
for spk in tqdm.tqdm(os.listdir(base_dir)):
    get_speak_emb(spk)
    
np.save('/home/xingxing/COMP5214/tts_ref/Speech-Backbones/Grad-TTS-parallel/resources/filelists/esd/spk_emb.npy', spk_emb_dict)