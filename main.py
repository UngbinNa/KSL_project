import logging
import os
import shutil
import time
from itertools import groupby
from torch import from_numpy, tensor
# import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pdb
import tensorflow as tf
from lib.config import get_cfg
from lib.dataset import build_data_loader
from lib.engine import default_argument_parser, default_setup
from lib.model import SignModel
from lib.solver import build_lr_scheduler, build_optimizer
from lib.utils import AverageMeter, clean_ksl, wer_list
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 자동설정 안될경우
else:
    print("CUDA is not available. Running on CPU.")

#tf.config.set_visible_devices([], "GPU")
best_wer = 100


def setup(args):
    """
    Create configs and perform basic setups.
    """
    
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg = default_setup(cfg, args)
    return cfg




def main(args):
    global best_wer
    logger = logging.getLogger()

    EPOCHS = 80
    start_epoch = 0
    cfg = setup(args)
    cfg.freeze()
    train_loader, val_loader = build_data_loader(cfg)


    # loss_gls = nn.CTCLoss(blank=train_loader.dataset.sil_idx, zero_infinity=True).cuda()
    loss_gls = nn.CTCLoss(blank=train_loader.dataset.sil_idx, zero_infinity=True).to(cfg.GPU_ID)
    model = SignModel(train_loader.dataset.vocab)
    # model = model.cuda()
    model = model.to(cfg.GPU_ID)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    
    if cfg.RESUME:
        assert os.path.isfile(cfg.RESUME), "Error: no checkpoint directory found!"
        #checkpoint = torch.load(cfg.RESUME)
        checkpoint = torch.load(cfg.RESUME, map_location=torch.device(cfg.GPU_ID))
        best_wer=checkpoint['best_wer']
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint['state_dict'])
        # model = nn.DataParallel(model).cuda()
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Loaded checkpoint from {}.  "
            "start_epoch: {cp[epoch]} current_lr: {lr:.5f}  "
            "recoded WER: {cp[wer]:.3f} (best: {cp[best_wer]:.3f})".format(
                cfg.RESUME, cp=checkpoint, lr=current_lr
            )
        )
    else:
        # model = nn.DataParallel(model).cuda()
        #model = model.cuda()******************************
        model = model.to(cfg.GPU_ID) # model=model.cuda()룰 일관성 유지용으로 변경함
    

    if args.eval_only:
        metricss=validate(cfg,model, val_loader, loss_gls)
        print("valildation loss: {metricss[loss]:.3f}  validation WER: {metricss[wer]:.3f}  ".format(metricss=metricss))
        return
    writer = SummaryWriter(cfg.OUTPUT_DIR)
    data_time_meter = AverageMeter()
    loss_meter = AverageMeter()
    iter_time_meter = AverageMeter()
    epoch_time_meter = AverageMeter()

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        # epoch start
        epoch_start = time.perf_counter()
        loader_iter = iter(train_loader)
        print("best_wer = ", best_wer)
        for _iter in range(len(train_loader)):
            start = time.perf_counter()
            (videos, video_lengths), (glosses, gloss_lengths) = next(loader_iter)
            videos = videos.to(cfg.GPU_ID)
            video_lengths = video_lengths.to(cfg.GPU_ID)
            glosses = glosses.to(cfg.GPU_ID)
            gloss_lengths = gloss_lengths.to(cfg.GPU_ID)
            """
            videos = videos.cuda()
            video_lengths = video_lengths.cuda()
            glosses = glosses.cuda()
            gloss_lengths = gloss_lengths.cuda()
            """
            data_time = time.perf_counter() - start
            data_time_meter.update(data_time, n=videos.size(0))

            gloss_scores = model(videos)  # (B, T, C)
            gloss_probs = gloss_scores.log_softmax(2).permute(1, 0, 2)  # (T, B, C)

            loss = loss_gls(gloss_probs, glosses, video_lengths.long() // 4, gloss_lengths.long())
            loss_meter.update(loss.item(), n=videos.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            current_lr = optimizer.param_groups[0]["lr"]
            iter_time = time.perf_counter() - start
            iter_time_meter.update(iter_time, n=videos.size(0))

            # print logs
            if (_iter > 0 and  # noqa
                ((_iter + 1) % cfg.PERIODS.LOG_ITERS == 0)):
                # write log
                current_iter = epoch * len(train_loader) + _iter
                logger.info(
                    "epoch: {}/{}, iter: {}/{}  "
                    "loss: {loss.val:.3f} (avg: {loss.avg:.3f})  "
                    "iter_time: {iter_time.val:.3f} (avg: {iter_time.avg:.3f})  "
                    "data_time: {data_time.val:.3f} (avg: {data_time.avg:.3f})  "
                    "lr: {lr:.5f}".format(
                        epoch + 1,
                        EPOCHS,
                        _iter + 1,
                        len(train_loader),
                        loss=loss_meter,
                        iter_time=iter_time_meter,
                        data_time=data_time_meter,
                        lr=current_lr
                    )
                )

                writer.add_scalar("misc/data_time", data_time_meter.avg, current_iter)
                writer.add_scalar("misc/iter_time", iter_time_meter.avg, current_iter)
                writer.add_scalar("train/loss", loss_meter.avg, current_iter)
                writer.add_scalar("misc/lr", current_lr, current_iter)

                data_time_meter.reset()
                iter_time_meter.reset()
                loss_meter.reset()

        # end of epoch
        scheduler.step()

        epoch_time = time.perf_counter() - epoch_start
        epoch_time_meter.update(epoch_time, n=1)

        remain = EPOCHS - (epoch + 1)
        writer.add_scalar("misc/eta", remain * epoch_time_meter.avg, epoch)

        # validate
        metrics = validate(cfg, model, val_loader, loss_gls)
        for k, v in metrics.items():
            writer.add_scalar("val/" + k, v, epoch)

        logger.info(
            "epoch: {}/{}  "
            "valildation loss: {metrics[loss]:.3f}  validation WER: {metrics[wer]:.3f}  ".format(
                epoch + 1, EPOCHS, metrics=metrics
            )
        )
        print()
        print()

        # checkpoint
        model_to_save = model.module if hasattr(model, "module") else model
        is_best = metrics["wer"] < best_wer
        best_wer = min(best_wer, metrics["wer"])
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'wer': metrics["wer"],
                'best_wer': best_wer,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, cfg.OUTPUT_DIR
        )


def validate(cfg, model, val_loader, criterion) -> dict:
    logger = logging.getLogger()

    model.eval()
    val_loss_meter = AverageMeter()

    all_glosses = []
    loader_iter = iter(val_loader)
    vocab = decoder = val_loader.dataset.vocab
    decoder = vocab.arrays_to_sentences
    glosses_gt = val_loader.dataset.gloss
    for _iter in range(len(val_loader)):
        with torch.no_grad():
            (videos, video_lengths), (glosses, gloss_lengths) = next(loader_iter)
            videos = videos.to(cfg.GPU_ID)
            video_lengths = video_lengths.to(cfg.GPU_ID)
            glosses = glosses.to(cfg.GPU_ID)
            gloss_lengths = gloss_lengths.to(cfg.GPU_ID)
            """
            videos = videos.cuda()
            video_lengths = video_lengths.cuda()
            glosses = glosses.cuda()
            gloss_lengths = gloss_lengths.cuda()
            """
            #videos=torch.ones_like(videos)
            gloss_scores = model(videos)  # (B, T, C)
            #print(gloss_scores)
            gloss_probs = gloss_scores.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            
            loss = criterion(gloss_probs, glosses, video_lengths.long() // 4, gloss_lengths.long())
            val_loss_meter.update(loss, n=videos.size(0))
            # log loss

            # detach
            gloss_probs = gloss_probs.cpu().detach().numpy()  # (T, B, C)
            gloss_probs_tf = np.concatenate(
                # (C: 1~)
                (gloss_probs[:, :, 1:], gloss_probs[:, :, 0, None]),
                axis=-1,
            )
            sequence_length = video_lengths.long().cpu().detach().numpy() // 4
            ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
                inputs=gloss_probs_tf,
                sequence_length=sequence_length,
                beam_width=1,
                top_paths=1,
            )
            ctc_decode = ctc_decode[0]

            # Create a decoded gloss list for each sample
            tmp_gloss_sequences = [[] for i in range(gloss_scores.shape[0])]  # (B, )
            for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
                tmp_gloss_sequences[dense_idx[0]].append(ctc_decode.values[value_idx].numpy() + 1)
            decoded_gloss_sequences = []
            for seq_idx in range(0, len(tmp_gloss_sequences)):
                decoded_gloss_sequences.append(
                    [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
                )
            all_glosses.extend(decoded_gloss_sequences)
            if (_iter + 1) % 100 == 0:
                logger.info(
                    "valid iter: {}/{}  "
                    "loss: {loss.val:.3f} (avg: {loss.avg:.3f})  ".format(
                        _iter + 1,
                        len(val_loader),
                        loss=val_loss_meter,
                    )
                )
                logger.info("---------------------------------------------")

                decoded = decoder(arrays=decoded_gloss_sequences)
                decoded_gt = decoder(arrays=glosses.cpu().detach().numpy())
                for (_dec, _ref2) in zip(decoded[:4], decoded_gt[:4]):
                    _ref2 = [gloss for gloss in _ref2 if gloss != vocab.pad_token]
                    logger.info(" ".join(_dec))
                    logger.info(" ".join(_ref2))
                    print()

    assert len(all_glosses) == len(val_loader.dataset)
    decoded_gls = val_loader.dataset.vocab.arrays_to_sentences(arrays=all_glosses)
    # Gloss clean-up function
    
    # Construct gloss sequences for metrics
    gls_ref = [clean_ksl(" ".join(t)) for t in glosses_gt]
    gls_hyp = [clean_ksl(" ".join(t)) for t in decoded_gls]

    assert len(gls_ref) == len(gls_hyp)
    
    # GLS Metrics
    metrics = wer_list(hypotheses=gls_hyp, references=gls_ref)
    metrics.update({"loss": val_loss_meter.avg})
    
    return metrics


def save_checkpoint(
    state_dict: dict, is_best: bool, checkpoint: str, filename='checkpoint.pth.tar'
):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state_dict, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


if __name__ == "__main__":
    
    args = default_argument_parser().parse_args()
    #args.config_file='configs/config.yaml'
    main(args)
