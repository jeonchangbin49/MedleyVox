import argparse

import torch
import torch.nn as nn
import torchaudio
import fairseq


""" Wav2Vec 2.0 """


class Wav2Vec2FeatureExtractor(nn.Module):
    def __init__(
        self,
        checkpoint_dir,
        sr,
        model_version="xlsr",
        using_feature="c",
        extraction_method="none",
        device="cpu",
    ):
        super().__init__()
        """
        install requirements for Wav2Vec 2.0 feature extractor ###
        $ git clone https://github.com/pytorch/fairseq
        $ cd fairseq
        $ pip install --editable ./
        """
        """
        Unfortunately, this installation will not work cleanly...
        You first have to modify some code in fairseq, then 'pip install .....'
        This needs some googling.
        """

        self.device = device

        self.using_feature = using_feature
        self.extraction_method = extraction_method

        # set checkpoint dir
        if model_version.lower() == "xlsr":
            model_pt_name = "xlsr_53_56k.pt"
        ckpt_path = f"{checkpoint_dir}/{model_pt_name}"

        # set using feature : rather to use 'z' or 'c' feature
        # reload model
        (
            wav2vec_model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        wav2vec_model = wav2vec_model[0]

        # send to GPU
        self.model = wav2vec_model.to(self.device)

        # we need to resample the signal to use it as an input of wav2vec2.0
        ori_freq = sr
        new_freq = 16000

        # resampler
        self.resample_func = (
            torchaudio.transforms.Resample(orig_freq=ori_freq, new_freq=new_freq).to(
                self.device
            )
            if ori_freq != new_freq
            else None
        )
        self.resampling_ratio = new_freq / ori_freq

    def forward(
        self, audio_data, nth_layer_output=None
    ):  # nth_layer_output is LIST format (ex: [1,13])
        # resample if required and set expected output length
        if self.resample_func:
            audio_data = self.resample_func(audio_data)
        # receptive field == 400 samples
        # stride = 320 samples
        # audio_length = (audio_length.type(torch.IntTensor) - 80) // 320

        # inference
        self.model.eval()
        with torch.no_grad():
            if self.using_feature == "z":
                feature = self.model.feature_extractor(audio_data)
                # time_dim = 2
                # feat_dim = 1
            elif self.using_feature == "c":
                if nth_layer_output:
                    feature = self.model(
                        audio_data,
                        features_only=True,
                        mask=False,
                        layer=nth_layer_output[-1],
                    )
                    stacked_layers = []
                    for layer in nth_layer_output:
                        stacked_layers.append(feature["layer_results"][layer][0])
                    feature = torch.cat(stacked_layers, dim=2)  # (time, batch, feature)
                    feature = torch.transpose(feature, 0, 1)  # (batch, time, feature)
                    # time_dim=1
                    # feat_dim=2
                else:
                    feature = self.model(audio_data, features_only=True, mask=False)[
                        "x"
                    ]
                    # time_dim = 1
                    # feat_dim = 2

        return feature


# check feature extractor
if __name__ == "__main__":
    """
    Test code for Wav2Vec feature extractor
    """
    parser = argparse.ArgumentParser(description="model test.py")

    parser.add_argument("--target", type=str, default="vocals")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--inference_data_dir", type=str, default="./segments/24k")
    parser.add_argument("--results_save_dir", type=str, default="./my_sep_results")

    args, _ = parser.parse_known_args()

    # input setup
    batch_size = 1
    max_len = 534
    input_length = [max_len for i in range(batch_size)]

    # pad audio
    audio_list = []
    for cur_input_len in input_length:
        audio_list.append(torch.rand(cur_input_len))
    from torch.nn.utils.rnn import pad_sequence

    padded_aud = pad_sequence(audio_list, batch_first=True)

    ckpt_dir = "/path/to/w2v/"
    """ Wav2Vec """
    print("checking Wav2Vec feature extractor..")
    # xlsr
    wav2vec_fe = Wav2Vec2FeatureExtractor(
        args=args,
        checkpoint_dir=ckpt_dir,
        using_feature="c",
        model_version="xlsr",
        extraction_method="none",
    )
    output = wav2vec_fe(padded_aud.cuda(), input_length, nth_layer_output=[1, 13])
