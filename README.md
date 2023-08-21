# ZETTS

Official implementation of the ZETTS model based on [Grad-TTS](https://arxiv.org/abs/2105.06337).


Demo page: [https://iooops.github.io/cgrad/](https://iooops.github.io/cgrad/)


## Installation

Firstly, install all Python package requirements:

```bash
pip install -r requirements.txt
```

Secondly, build `monotonic_align` code (Cython):

```bash
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

**Note**: code is tested on Python==3.6.9.


## Inference

Download pretrained HifiGAN universal model and trained ZETTS model to folder **checkpts**: 

HifiGAN universal model: [https://drive.google.com/drive/folders/1YuOoV3lO2-Hhn1F2HJ2aQ4S0LC1JdKLd](https://drive.google.com/drive/folders/1YuOoV3lO2-Hhn1F2HJ2aQ4S0LC1JdKLd)

ZETTS model: [https://drive.google.com/file/d/1hkveKTdubcmHXN_esYhyl175Dpc3yCnB/view?usp=sharing](https://drive.google.com/file/d/1hkveKTdubcmHXN_esYhyl175Dpc3yCnB/view?usp=sharing)


Then run: 
```bash
bash infer.sh
```

## Train

Download ESD dataset: [https://github.com/HLTSingapore/Emotional-Speech-Data](https://github.com/HLTSingapore/Emotional-Speech-Data)


Refer to **prepare_esd_dataset.ipynb** and **prepare_speaker_emb.ipynb** to prepare ESD dataset, get speaker embeddings and emotion embeddings. Or you may use the processed files in **resources/filelists/esd**.


Then run:
```bash
bash train.sh
```

(The default mode is multi-GPU training)




