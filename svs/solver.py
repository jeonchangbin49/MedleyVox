import time
import json

import torch
import torch.nn as nn
from asteroid.losses import (
    PITLossWrapper,
    pairwise_neg_sisdr,
    pairwise_neg_sdsdr,
    PairwiseNegSDR,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import wandb
from ema_pytorch import EMA

from . import utils
from .models import load_model_with_args
from .losses import (
    SingleSrcMultiScaleSpectral_modified,
    SingleSrcMultiScaleSpectral_TRUnet,
    SingleSrcMultiScaleSpectral_TRUNet_freq,
    SingleSrcMultiScaleSpectral_L1,
    SingleSrcMultiScaleSpectral_L1_above_freq,
)
from .data import (
    DuetSingingSpeechMixTraining,
    DuetSingingSpeechMixValidation,
    MultiSingingSpeechMixTraining,
    MultiSingingSpeechMixValidation,
)


class Solver(object):
    def __init__(self):
        pass

    def set_gpu(self, args):
        if args.use_wandb and args.gpu == 0:
            if args.sweep:
                wandb.init(
                    entity=args.entity,
                    project=args.project,
                    config=args,
                    resume=True if args.resume != None and args.gpu == 0 else False,
                )
            else:
                wandb.init(
                    entity=args.entity,
                    project=args.project,
                    name=f"{args.exp_name}_{args.gpu}",
                    group="singing_sep",
                    config=args,
                    resume=False,
                    settings=wandb.Settings(start_method="fork"),
                )

        ###################### Define Models ######################
        trainable_params = []
        # load model
        self.model = load_model_with_args(args)
        if args.mixture_consistency == "sfsrnet":
            total_params = list(self.model.masker.parameters())
            sr_net_params = list(self.model.sr_net.parameters())
            trainable_params = [
                {"params": sr_net_params, "lr": args.lr},
                {"params": total_params, "lr": args.lr},
            ]
        else:
            trainable_params = trainable_params + list(self.model.parameters())

        if args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                params=trainable_params,
                lr=args.lr,
                momentum=0.9,
                eps=args.eps,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                params=trainable_params,
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
                amsgrad=False,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer == "radam":
            self.optimizer = torch.optim.RAdam(
                params=trainable_params,
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                params=trainable_params,
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
                weight_decay=args.weight_decay,
            )
        else:
            print("no optimizer loaded")
            raise NotImplementedError

        if args.lr_scheduler == "step_lr":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=args.lr_decay_gamma,
                patience=args.lr_decay_patience,
                cooldown=0,
                min_lr=5e-5,
                verbose=True,
            )
        elif args.lr_scheduler == "cos_warmup":
            self.scheduler = utils.CosineAnnealingWarmUpRestarts(
                self.optimizer, T_0=40, T_mult=1, eta_max=args.lr, T_up=10, gamma=0.5
            )

        torch.cuda.set_device(args.gpu)

        self.model = self.model.to(f"cuda:{args.gpu}")

        ############################################################
        # Define Loss
        self.criterion = {}
        self.criterion["L1"] = nn.L1Loss().to(args.gpu)
        self.criterion["si_sdr"] = pairwise_neg_sisdr.to(args.gpu)
        self.criterion["snr"] = PairwiseNegSDR("snr", EPS=args.eps).to(args.gpu)
        self.criterion["mse"] = nn.MSELoss().to(args.gpu)
        self.criterion["bce"] = nn.BCEWithLogitsLoss().to(args.gpu)

        if args.mixed_precision:
            pairwise_neg_sisdr_ = PairwiseNegSDR("sisdr", EPS=args.eps)
            self.criterion["pit_si_sdr"] = PITLossWrapper(
                pairwise_neg_sisdr_, pit_from="pw_mtx"
            ).to(args.gpu)
            self.criterion["pit_sd_sdr"] = PITLossWrapper(
                pairwise_neg_sisdr_, pit_from="pw_mtx"
            ).to(args.gpu)

            pairwise_neg_snr_ = PairwiseNegSDR("snr", EPS=args.eps)
            self.criterion["pit_snr"] = PITLossWrapper(
                pairwise_neg_snr_, pit_from="pw_mtx"
            ).to(args.gpu)

        else:
            self.criterion["pit_si_sdr"] = PITLossWrapper(
                pairwise_neg_sisdr, pit_from="pw_mtx"
            ).to(args.gpu)
            self.criterion["pit_sd_sdr"] = PITLossWrapper(
                pairwise_neg_sdsdr, pit_from="pw_mtx"
            ).to(args.gpu)

            pairwise_neg_snr_ = PairwiseNegSDR("snr")
            self.criterion["pit_snr"] = PITLossWrapper(
                pairwise_neg_snr_, pit_from="pw_mtx"
            ).to(args.gpu)

        # Spectral loss
        self.criterion["multi_spectral_l1"] = SingleSrcMultiScaleSpectral_L1(
            loss_scale=100.0,
            log_scale=args.multi_spec_loss_log_scale,
        ).to(args.gpu)
        self.criterion[
            "multi_spectral_l1_above_freq"
        ] = SingleSrcMultiScaleSpectral_L1_above_freq(
            loss_scale=100.0,
            sample_rate=args.sample_rate,
            above_freq=args.above_freq,
            log_scale=args.multi_spec_loss_log_scale,
        ).to(
            args.gpu
        )

        self.criterion["pit_multi_spectral_l1"] = PITLossWrapper(
            self.criterion["multi_spectral_l1"], pit_from="pw_pt"
        ).to(args.gpu)
        self.criterion["multi_spectral"] = SingleSrcMultiScaleSpectral_modified().to(
            args.gpu
        )
        self.criterion["multi_spectral_trunet"] = SingleSrcMultiScaleSpectral_TRUnet(
            loss_scale=0.01,
            log_scale=args.multi_spec_loss_log_scale,
        ).to(args.gpu)
        self.criterion[
            "multi_spectral_trunet_above_freq"
        ] = SingleSrcMultiScaleSpectral_TRUNet_freq(
            loss_scale=0.01,
            sample_rate=args.sample_rate,
            above_freq=args.above_freq,
            log_scale=args.multi_spec_loss_log_scale,
        ).to(
            args.gpu
        )

        self.es = utils.EarlyStopping(patience=args.patience)
        self.stop = False

        if args.use_wandb and args.gpu == 0:
            wandb.watch(self.model, log="all")

        self.start_epoch = 1
        self.train_losses = []
        self.valid_losses = []
        self.train_times = []
        self.best_epoch = 0

        if args.resume and not args.ema:
            self.resume(args)

        # Distribute models to machine
        self.model = DDP(
            self.model,
            device_ids=[args.gpu],
            output_device=args.gpu,
            find_unused_parameters=True,
        )

        if args.ema:
            self.model_ema = EMA(
                self.model,
                beta=0.999,
                update_after_step=100,
                update_every=10,
            )

        if args.resume and args.ema:
            self.resume(args)

        ###################### Define data pipeline ######################
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        self.mp_context = torch.multiprocessing.get_context("fork")
        if args.dataset == "singing_librispeech":
            self.train_dataset = DuetSingingSpeechMixTraining(
                singing_data_dir=args.train_root,
                speech_data_dir=args.speech_train_root,
                song_length_dict_path=args.song_length_dict_path,
                same_song_dict_path=args.same_song_dict_path,
                same_singer_dict_path=args.same_singer_dict_path,
                same_speaker_dict_path=args.same_speaker_dict_path,
                sample_rate=args.sample_rate,
                segment=args.seq_dur,
                unison_prob=args.unison_prob,
                pitch_formant_augment_prob=args.pitch_formant_augment_prob,
                augment=True,
                part_of_data=args.part_of_data,
                reduced_training_data_ratio=args.reduced_training_data_ratio,
                sing_sing_ratio=args.sing_sing_ratio,
                sing_speech_ratio=args.sing_speech_ratio,
                same_song_ratio=args.same_song_ratio,
                same_singer_ratio=args.same_singer_ratio,
                same_speaker_ratio=args.same_speaker_ratio,
                # speech_speech_ratio=args.speech_speech_ratio
            )
            self.valid_dataset = []
            for valid_data_path_list in args.valid_root:
                self.valid_dataset.append(
                    DuetSingingSpeechMixValidation(
                        data_dir=[valid_data_path_list],
                        sample_rate=args.sample_rate,
                        segment=args.seq_dur,
                        augment=True,
                    )
                )
        elif args.dataset == "multi_singing_librispeech":
            self.train_dataset = MultiSingingSpeechMixTraining(
                singing_data_dir=args.train_root,
                speech_data_dir=args.speech_train_root,
                song_length_dict_path=args.song_length_dict_path,
                same_song_dict_path=args.same_song_dict_path,
                same_singer_dict_path=args.same_singer_dict_path,
                same_speaker_dict_path=args.same_speaker_dict_path,
                min_n_src=args.min_n_src,
                max_n_src=args.max_n_src,
                sample_rate=args.sample_rate,
                segment=args.seq_dur,
                unison_prob=args.unison_prob,
                pitch_formant_augment_prob=args.pitch_formant_augment_prob,
                augment=True,
                part_of_data=args.part_of_data,
                reduced_training_data_ratio=args.reduced_training_data_ratio,
                sing_sing_ratio=args.sing_sing_ratio,
                sing_speech_ratio=args.sing_speech_ratio,
                same_song_ratio=args.same_song_ratio,
                same_singer_ratio=args.same_singer_ratio,
                same_speaker_ratio=args.same_speaker_ratio,
                # speech_speech_ratio=args.speech_speech_ratio
            )
            self.valid_dataset = []
            for valid_data_path_list in args.valid_root_orpit:
                self.valid_dataset.append(
                    MultiSingingSpeechMixValidation(
                        data_dir=[valid_data_path_list],
                        sample_rate=args.sample_rate,
                        segment=args.seq_dur,
                        augment=True,
                    )
                )

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, rank=args.gpu
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.nb_workers,
            multiprocessing_context=self.mp_context,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=False,
            worker_init_fn=utils.worker_init_fn,
            persistent_workers=True,
        )
        self.valid_sampler = []
        self.valid_loader = []
        for i, valid_dataset_ in enumerate(self.valid_dataset):
            self.valid_sampler.append(
                DistributedSampler(valid_dataset_, shuffle=False, rank=args.gpu)
            )
            self.valid_loader.append(
                torch.utils.data.DataLoader(
                    valid_dataset_,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.nb_workers,
                    multiprocessing_context=self.mp_context,
                    pin_memory=False,
                    sampler=self.valid_sampler[i],
                    drop_last=False,
                )
            )

        if args.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

    def train(self, args, epoch):
        self.end = time.time()
        self.model.train()
        self.train_sampler.set_epoch(epoch)

        # get current learning rate
        for param_group in self.optimizer.param_groups:
            current_lr = param_group["lr"]

        if (
            args.rank % args.ngpus_per_node == 0
        ):  # when the last rank process is finished
            print(f"Epoch {epoch}, Learning rate: {current_lr}")

        losses = utils.AverageMeter()
        loss_logger = {}

        loss_logger["epoch-wise-train/train loss"] = 0
        # with torch.autograd.detect_anomaly(): # use this if you want to detect anomaly behavior while training.
        for i, values in enumerate(self.train_loader):
            mixture, clean = values

            mixture = mixture.cuda(args.gpu, non_blocking=True)
            clean = clean.cuda(args.gpu, non_blocking=True)

            target = clean

            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    if args.mixture_consistency == "sfsrnet":
                        estimates = self.model.module.forward_pre(mixture)
                    else:
                        estimates = self.model(mixture)
            else:
                if args.mixture_consistency == "sfsrnet":
                    estimates = self.model.module.forward_pre(mixture)
                else:
                    estimates = self.model(mixture)

            loss_input = {}

            if (
                len(args.train_loss_func) == 1
            ):  # train_loss_function only uses pit_si_sdr (or pit_other) loss
                loss_input[args.train_loss_func[0]] = (estimates, target)

                loss_dict = self.cal_loss(args, loss_input)
                loss = sum([value.mean() for key, value in loss_dict.items()])

            else:  # train_loss_function uses other losses
                loss = []
                for train_loss_idx, single_train_loss_func in enumerate(
                    args.train_loss_func
                ):
                    # first single_train_loss_func should be 'pit_si_sdr or pit_sd_sdr' or 'pit_snr' or 'pit_multi_spectral_l1'
                    if single_train_loss_func == (
                        "pit_si_sdr"
                        or "pit_sd_sdr"
                        or "pit_snr"
                        or "pit_multi_spectral_l1"
                    ):
                        loss_pit, estimates = self.criterion[
                            single_train_loss_func
                        ].forward(estimates, target, return_est=True)
                        loss.append(loss_pit.mean())
                    else:
                        if (
                            args.mixture_consistency == "sfsrnet"
                            and len(args.train_loss_func) == 2
                        ):
                            estimates = self.model.module.forward_sr(
                                mixture, estimates
                            )  # for example, when using --train_loss_func pit_si_sdr si_sdr
                        elif (
                            args.mixture_consistency == "sfsrnet"
                            and len(args.train_loss_func) >= 3
                        ):  # --train_loss_func pit_snr multi_spectral_l1 snr multi_spectral_l1
                            if train_loss_idx == 2:
                                estimates = self.model.module.forward_sr(
                                    mixture, estimates
                                )  # for example
                            else:
                                pass
                        loss_else = self.criterion[single_train_loss_func](
                            estimates, target
                        )
                        loss.append(loss_else.mean())
                loss = sum(loss)
            ############################################################

            #################### 5. Back propagation ####################
            if args.mixed_precision:
                self.scaler.scale(loss).backward()
                if args.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=args.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if args.gradient_clip:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=args.gradient_clip
                    )
                self.optimizer.step()

            losses.update(loss.item(), clean.size(0))

            loss_logger["epoch-wise-train/train loss"] = loss.item()

            self.model.zero_grad()

            if args.ema:
                self.model_ema.update()
            ############################################################

            # ###################### 6. Plot ######################
            if i % 30 == 0:
                # loss print for multiple loss function
                multiple_score = torch.Tensor(
                    [value for key, value in loss_logger.items()]
                ).to(args.gpu)
                gathered_score_list = [
                    torch.ones_like(multiple_score)
                    for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_score_list, multiple_score)
                gathered_score = torch.mean(
                    torch.stack(gathered_score_list, dim=0), dim=0
                )
                if args.gpu == 0:
                    print(f"Epoch {epoch},  step {i} / {len(self.train_loader)}")
                    temp_loss_logger = {}
                    for index, (key, value) in enumerate(loss_logger.items()):
                        temp_key = key.replace("epoch-wise-train/", "iter-wise/")
                        temp_loss_logger[temp_key] = round(
                            gathered_score[index].item(), 6
                        )
                        print(f"{key} : {round(gathered_score[index].item(), 6)}")

        single_score = torch.Tensor([losses.avg]).to(args.gpu)
        loss_logger["epoch-wise-train/train loss"] = single_score

        if args.use_wandb and args.gpu == 0:
            loss_logger["epoch-wise-train/epoch"] = epoch
            wandb.log(loss_logger)

        gathered_score_list = [
            torch.ones_like(single_score) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_score_list, single_score)
        gathered_score = torch.mean(torch.cat(gathered_score_list)).item()
        if args.gpu == 0:
            self.train_losses.append(gathered_score)

    def multi_validate(self, args, epoch):
        if args.gpu == 0:
            print(f"Epoch {epoch} Validation session!")

        losses = utils.AverageMeter()

        loss_logger = {}

        self.model.eval()
        with torch.no_grad():
            valid_loss_accumulated = 0
            for valid_loader_idx, valid_loader in enumerate(self.valid_loader):
                if args.dataset == "multi_singing_librispeech":
                    valid_loader_category = args.valid_root_orpit[valid_loader_idx][3]
                else:
                    valid_loader_category = args.valid_root[valid_loader_idx][3]
                for i, values in enumerate(valid_loader, start=1):

                    mixture, clean = values

                    mixture = mixture.cuda(args.gpu, non_blocking=True)
                    clean = clean.cuda(args.gpu, non_blocking=True)

                    target = clean

                    loss_input = {}

                    if args.mixed_precision:
                        with torch.cuda.amp.autocast():
                            if args.ema:
                                estimates = self.model_ema(mixture)
                            else:
                                estimates = self.model(mixture)
                    else:
                        if args.ema:
                            estimates = self.model_ema(mixture)
                        else:
                            estimates = self.model(mixture)

                    if (
                        args.architecture == "conv_tasnet_stft"
                        or args.architecture == "conv_tasnet_learnable_basis"
                    ):
                        if len(args.valid_loss_func) == 1:
                            loss_input[args.valid_loss_func[0]] = (estimates, target)

                            loss_dict = self.cal_loss(
                                args, loss_input
                            )  # 각 loss 에 대한 계산. loss 를 dict 로 저장
                            loss = sum(
                                [value.mean() for key, value in loss_dict.items()]
                            )
                        else:
                            loss = []
                            for single_valid_loss_func in args.valid_loss_func:
                                if single_valid_loss_func == (
                                    "pit_si_sdr" or "pit_sd_sdr"
                                ):
                                    loss_pit, estimates = self.criterion[
                                        single_valid_loss_func
                                    ].forward(estimates, target, return_est=True)
                                    loss.append(loss_pit)
                                else:
                                    loss_else = self.criterion[single_valid_loss_func](
                                        estimates, target
                                    )
                                    loss.append(loss_else)
                            loss = sum(loss)
                        losses.update(loss.item(), clean.size(0))
                    else:
                        if len(args.valid_loss_func) == 1:

                            loss_input[args.valid_loss_func[0]] = (estimates, target)

                            loss_dict = self.cal_loss(
                                args, loss_input
                            )  # 각 loss 에 대한 계산. loss 를 dict 로 저장
                            loss = sum([value for key, value in loss_dict.items()])
                        else:
                            loss = []
                            for single_valid_loss_func in args.valid_loss_func:
                                if single_valid_loss_func == (
                                    "pit_si_sdr" or "pit_sd_sdr"
                                ):
                                    loss_pit, estimates = self.criterion[
                                        single_valid_loss_func
                                    ].forward(estimates, target, return_est=True)
                                    loss.append(loss_pit)
                                else:
                                    loss_else = self.criterion[single_valid_loss_func](
                                        estimates, target
                                    )
                                    loss.append(loss_else)
                            loss = sum(loss)
                        losses.update(loss.item(), clean.size(0))

                single_score = torch.Tensor([losses.avg]).to(args.gpu)
                gathered_score_list = [
                    torch.ones_like(single_score) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_score_list, single_score)
                gathered_score = torch.mean(torch.cat(gathered_score_list)).item()

                loss_logger[
                    f"epoch-wise-valid/{valid_loader_category} valid loss"
                ] = gathered_score

                valid_loss_accumulated = valid_loss_accumulated + gathered_score

                if args.use_wandb and args.gpu == 0:
                    loss_logger["epoch-wise-valid/epoch"] = epoch
                    wandb.log(loss_logger)

            valid_loss_accumulated = valid_loss_accumulated / len(self.valid_loader)

            if args.lr_scheduler == "step_lr":
                self.scheduler.step(valid_loss_accumulated)
            elif args.lr_scheduler == "cos_warmup":
                self.scheduler.step(epoch)
            else:
                self.scheduler.step(valid_loss_accumulated)

            if args.gpu == 0:
                self.valid_losses.append(valid_loss_accumulated)

                self.stop = self.es.step(valid_loss_accumulated)

                print(
                    f"Epoch {epoch}, validation loss : {round(valid_loss_accumulated, 6)}"
                )

                plt.plot(self.train_losses, label="train loss")
                plt.plot(self.valid_losses, label="valid loss")
                plt.legend(loc="upper right")
                plt.savefig(f"{args.output}/loss_graph_{args.target}.png")
                plt.close()

                save_states = {
                    "epoch": epoch,
                    "state_dict": self.model.module.state_dict()
                    if not args.ema
                    else self.model_ema.state_dict(),
                    "best_loss": self.es.best,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                }

                utils.save_checkpoint(
                    save_states,
                    state_dict_only=valid_loss_accumulated == self.es.best,
                    path=args.output,
                    target=args.target,
                )

                self.train_times.append(time.time() - self.end)

                # 일단 pass
                if valid_loss_accumulated == self.es.best:
                    self.best_epoch = epoch

                # save params
                params = {
                    "epochs_trained": epoch,
                    "args": vars(args),
                    "best_loss": self.es.best,
                    "best_epoch": self.best_epoch,
                    "train_loss_history": self.train_losses,
                    "valid_loss_history": self.valid_losses,
                    "train_time_history": self.train_times,
                    "num_bad_epochs": self.es.num_bad_epochs,
                }

                with open(f"{args.output}/{args.target}.json", "w") as outfile:
                    outfile.write(json.dumps(params, indent=4, sort_keys=True))

                self.train_times.append(time.time() - self.end)
                print(
                    f"Epoch {epoch} train completed. Took {round(self.train_times[-1], 3)} seconds"
                )

    def resume(self, args):
        print(f"Resume checkpoint from: {args.resume}:")
        loc = f"cuda:{args.gpu}"
        checkpoint_path = f"{args.resume}/{args.target}"
        with open(f"{checkpoint_path}.json", "r") as stream:
            results = json.load(stream)
        checkpoint = torch.load(f"{checkpoint_path}.chkpnt", map_location=loc)
        if args.start_from_best:
            del checkpoint["state_dict"]
            checkpoint["state_dict"] = torch.load(
                f"{checkpoint_path}.pth", map_location=loc
            )
            print("start from best weight")
        if args.ema:
            if args.mixture_consistency == "sfsrnet":
                model_dict = self.model_ema.state_dict()
                # 1. filter out unnecessary keys
                checkpoint["state_dict"] = {
                    k: v for k, v in checkpoint["state_dict"].items() if k in model_dict
                }
                # 2. overwrite entries in the existing state dict
                model_dict.update(checkpoint["state_dict"])
                # 3. load the new state dict
                self.model_ema.load_state_dict(checkpoint["state_dict"], strict=False)
            else:
                self.model_ema.load_state_dict(checkpoint["state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            if args.load_ema_online_model:
                print("load ema online model!!")
                model_dict = self.model.state_dict()
                # 1. filter out unnecessary keys
                checkpoint["state_dict"] = {
                    k.replace("online_model.module.", ""): v
                    for k, v in checkpoint["state_dict"].items()
                    if k.replace("online_model.module.", "") in model_dict
                }
                # 2. overwrite entries in the existing state dict
                model_dict.update(checkpoint["state_dict"])
                # 3. load the new state dict
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint["state_dict"])

            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if (
            args.continual_train
        ):  # we want to use pre-trained model but not want to use lr_scheduler history.
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = args.lr
        else:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.es.best = results["best_loss"]
            self.es.num_bad_epochs = results["num_bad_epochs"]

        self.start_epoch = results["epochs_trained"] + 1
        self.train_losses = results["train_loss_history"]
        self.valid_losses = results["valid_loss_history"]
        self.train_times = results["train_time_history"]
        self.best_epoch = results["best_epoch"]

        if args.rank % args.ngpus_per_node == 0:
            print(
                f"=> loaded checkpoint {checkpoint_path} (epoch {results['epochs_trained']})"
            )

    def cal_loss(self, args, loss_input):
        loss_dict = {}
        for key, value in loss_input.items():
            loss_dict[key] = self.criterion[key](*value)

        return loss_dict
