# MedleyVox

This is an official page of "MedleyVox: An Evaluation Dataset for Multiple Singing Voices Separation" submitted to ICASSP 2023.

[![arXiv](https://img.shields.io/badge/arXiv-2211.07302-b31b1b.svg)](https://arxiv.org/abs/2211.07302)
[![Web](https://img.shields.io/badge/Web-Audio%20Samples-green.svg)](https://catnip-leaf-c6a.notion.site/Audio-Samples-of-MedleyVox-An-Evaluation-Dataset-for-Multiple-Singing-Voices-Separation-30074b2c88d24f46b68d9293f6095962)


### Notes

In the submitted version to ICASSP 2023, we had some mistakes on cIRM calculation and wrote some wrong statements related to that => we fixed them in arXiv preprint. Of course, we will fix them on our rebuttal phase. 

## How to obtain MedleyVox
We are going to upload MedleyVox data on Zenodo (or other platforms) after the ICASSP review process. For now, if you want to use our data for your experiments or application developments, you can now obtain it from the metadata of MedleyVox. You first have to download MedleyDB v1 and v2 to create MedleyVox dataset, then run the following code.

```
python -m testset.testset_save
```

## Training

### Preparation

Our code is heavily based on asteroid. You first have to install asteroid as a python package.

```
pip install git+https://github.com/asteroid-team/asteroid
```

and other remaining packages in ‘requirements.txt’. [fairseq](https://github.com/facebookresearch/fairseq) package is not needed for training but you need to use it when you use the chunk-wise processing based on wav2vec representation. It will be introduced in the last section of this page.

In svs/preprocess folder, you can find a number of preprocessing codes. For preparation of train data, almost of the codes are just simple downsampling/save processes. For preparation of validation data, you can ignore it because we already made json files of metadata for validation.

### Mixture construction strategy for training

For mixture construction strategy, we have a total of 5 arguments in svs/main.py for 6 training input construction strategy. Each of them are 


1. sing_sing_ratio (float) : Case 1. Ratio of 'different singing + singing' in training data sampling process. 
2. sing_speech_ratio (float) : Case 2. Ratio of 'different singing + speech' in training data sampling process. 
3. same_song_ratio (float) : Case 3. Ratio of 'same song of different singers’ in training data sampling process. 
4. same_singer_ratio (float) : Case 4. Ratio of 'different songs of same singer’ in training data sampling process. 
5. same_speaker_ratio (float) : Case 5. Ratio of 'different speeches of same speaker’ in training data sampling process. 
6. speech_speech_ratio (float) : Case 6. Ratio of 'different speech + speech’ in training data sampling process. This is not specified by arguments, but automatically calculated by ‘1 - (sum_of_rest_arguments)’. 


### Training details (duet and unison)

We first train the standard Conv-TasNet (for 200 epochs). 

```
python -m svs.main --exp_name=your_exp_name --patience=50\
--use_wandb=True --mixture_consistency=mixture_consistency\
--train_loss_func pit_snr multi_spectral_l1\
```

Then, we start joint training of the pre-trained Conv-TasNet and the cascaded iSRNet. (for 30 epochs with argument —reduced_training_data_ratio=0.1, for more frequent validation loss checking)

```
python -m svs.main --exp_name=your_exp_name_iSRNet\
--start_from_best=True --reduced_training_data_ratio=0.1\
--gradient_clip=5 --lr=3e-5 --batch_size=8 --above_freq=3000\
--epochs=230 --lr_decay_patience=6 --patience=15\
--use_wandb=True --mixture_consistency=sfsrnet --srnet=convnext\
--sr_input_res=False --train_loss_func pit_snr multi_spectral_l1 snr\
--continual_train=True --resume=/path/to/your_exp_name
```

### Training details (main_vs_rest)

Similar to duet and unison separation model, we first train the standard Conv-TasNet (for 200 epochs). You have to set different --dataset argument.

```
python -m svs.main --exp_name=your_exp_name --patience=50\
--use_wandb=True --mixture_consistency=mixture_consistency\
--train_loss_func pit_snr multi_spectral_l1\
--dataset=multi_singing_librispeech
```

After that, also similar to duet and unison separation model, we start joint training of the pre-trained Conv-TasNet and the cascaded iSRNet. (for 30 epochs with argument —reduced_training_data_ratio=0.1, for more frequent validation loss checking)

```
python -m svs.main --exp_name=your_exp_name_iSRNet\
--start_from_best=True --reduced_training_data_ratio=0.1\
--gradient_clip=5 --lr=3e-5 --batch_size=8 --above_freq=3000\
--epochs=230 --lr_decay_patience=6 --patience=15\
--use_wandb=True --mixture_consistency=sfsrnet --srnet=convnext\
--sr_input_res=False --train_loss_func pit_snr multi_spectral_l1 snr\
--continual_train=True --resume=/path/to/your_exp_name\
--dataset=multi_singing_librispeech
```

### Training dataset

We use a total of 13 different singing datasets of 400 hours and 460 hours of LibriSpeech data for training.

|Dataset|Labels (same song (segment) of different singers)|Labels (different songs of same singer)|Lengths[hours]|Notes|
|------|----------|-----------|-----|-----|
|[Children’s song dataset (CSD)](https://zenodo.org/record/4785016#.Y2-r2y_kFqs)|_|&check;|4.9|_|
|[NUS](https://drive.google.com/drive/folders/12pP9uUl0HTVANU3IPLnumTJiRjPtVUMx)|_|&check;|1.9|_|
|[TONAS](https://zenodo.org/record/1290722#.Y2-tci_kFqs)|_|_|0.3|_|
|[VocalSet](https://zenodo.org/record/1193957#.Y2-tmC_kFqs)|_|&check;|8.8|_|
|[Jsut-song](https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song)|_|&check;|0.4|_|
|[Jvs_music](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music)|_|&check;|2.3|_|
|[Tohoku Kiritan](https://zunko.jp/kiridev/login.php)|_|&check;|1.1|_|
|[vocadito](https://zenodo.org/record/5578807#.Y2-v2-zP1qs)|_|_|0.2|_|
|[Musdb-hq (train subset)](https://sigsep.github.io/datasets/musdb.html)|_|&check;|2.0|Single singing regions were extracted from the annotations in [musdb-lyrics extension](https://zenodo.org/record/3989267#.Y2-wBOzP1qs)|
|[OpenSinger](https://github.com/Multi-Singer/Multi-Singer.github.io)|_|&check;|51.9|_|
|[MedleyDB v1](https://medleydb.weebly.com/)|_|_|3.8|For training, we only used the songs that included in musdb18 dataset.|
|[K_multisinger](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=465)|&check;|&check;|169.6|_|
|[K_multitimbre](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=473)|&check;|&check;|150.8|_|
|[LibriSpeech_train-clean-360](http://www.openslr.org/12)|_|&check;|360|_|
|[LibriSpeech_train-clean-100](http://www.openslr.org/12)|_|&check;|100|_|


### Validation dataset

We use a musdb-hq (test subset) and LibriSpeech_dev-clean for validation data.

|Case|Description|Notes|
|-----|-----|-----|
|1)|Different singing + singing|—|
|2)|One singing + its unison|—|
|3)|Different songs of same singer|—|
|4)|Different speech + speech|—|
|5)|One speech + its unison|—|
|6)|Different speeches of same speaker|—|
|7)|Different speech + singing|—|


## How to test

Currently, we have no plan to upload the pre-trained weights of our models.

```
python -m svs.test --singing_task=duet --exp_name=your_exp_name
```

## How to Inference

separate every audio file (.mp3, .flac, .wav) in --inference_data_dir

```
python -m svs.inference --exp_name=your_exp_name\
--model_dir=/path/where/your/checkpoint/is\
--inference_data_dir=/path/where/the/input/data/is\
--results_save_dir=/path/to/save/output
```

### Chunk-wise processing

If the input is too long, it may be impossible to impossible due to lack of VRAM, or performance may be degraded at all. In that case, use --use_overlapadd. Among the --use_overlapadd options, "ola", "ola_norm", and "w2v" all work similarly to LambdaOverlapAdd in asteroid.

- ola: Same as LambdaOverlapAdd in asteroid.
- ola_norm: LambdaOverlapAdd with input applied chunk-wise loudness normalization (we used loudness normalization in training stage). The effect was not good. 
- w2v: When calculating the singer assignment in the overlapped region of the chunk in the LambdaOverlapAdd function based on the wave2vec2.0-xlsr model, the LambdaOverlapAdd implemented in the asteroid is simply obtained as L1 in the waveform stage. This is transformed into cosine similarity of w2v feature. You first have to install [fairseq](https://github.com/facebookresearch/fairseq) and download the weight of wav2vec2.0-xlsr model. 

In our paper, we have analyzed several failure cases that standard ola methods cannot handle. To this end, we implemented some useful inference methods for chunk-wise processing based on voice activity detection (VAD). 

- w2v_chunk: First use VAD and divide it into chunks, then chunk-wise processing. Unlike asteroid LambdaOverlapAdd, there is no overlapped region of chunk in front and rear, so it should not be implemented as L1 distance in waveform, and the similarity in feature stage is obtained. Calculated by continuously accumulating the w2v feature for each chunk.
- sf_chunk: The principle is the same as w2v_chunk, but instead of w2v, use a spectral feature such as mfcc or spectral centroid.

—vad_method can be used between spectrogram energy based (spec) and py-webrtcvad based (webrtc).
