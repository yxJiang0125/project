import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config.defaults import _C as config, update_config
from utils import train_util, log_util, loss_util, optimizer_util, anomaly_util
from models import Network_Myself
import datasets


def parse_args():
    parser = argparse.ArgumentParser(description='HSTA Training')
    parser.add_argument('--cfg', help='config file', default='config/avenue.yaml', type=str)
    parser.add_argument('opts', help='extra config overrides', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    update_config(config, args)

    logger, final_output_dir, _ = log_util.create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    gpus = list(config.GPUS)
    model = Network_Myself(config)
    logger.info('Model: {}'.format(model.get_name()))
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    losses = loss_util.MultiLossFunction(config=config).cuda()
    optimizer = optimizer_util.get_optimizer(config, model)
    scheduler = optimizer_util.get_scheduler(config, optimizer)

    train_dataset = datasets.get_data(config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True
    )
    logger.info('Number of training clips: {}'.format(len(train_dataset)))

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        train(config, train_loader, model, losses, optimizer, epoch, logger)
        scheduler.step()

        if (epoch + 1) % config.SAVE_CHECKPOINT_FREQ == 0:
            ckpt_path = os.path.join(final_output_dir, f'epoch_{epoch + 1}.pth')
            torch.save(model.module.state_dict(), ckpt_path)
            logger.info(f'Saved checkpoint: {ckpt_path}')

    final_state_path = os.path.join(final_output_dir, 'final_state.pth')
    torch.save(model.module.state_dict(), final_state_path)
    logger.info(f'Saved final model: {final_state_path}')


def train(config, train_loader, model, loss_functions, optimizer, epoch, logger):
    loss_func_mse = nn.MSELoss(reduction='none')
    model.train()

    for i, data in enumerate(train_loader):
        inputs, target = train_util.decode_input(input=data, train=True)
        output = model(inputs)

        target = target.cuda(non_blocking=True)
        inte_loss, grad_loss, msssim_loss, l2_loss = loss_functions(output, target)
        loss = inte_loss + grad_loss + msssim_loss + l2_loss

        mse_imgs = torch.mean(loss_func_mse((output + 1) / 2, (target + 1) / 2)).item()
        psnr = anomaly_util.psnr_park(mse_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % config.PRINT_FREQ == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            msg = (f'Epoch [{epoch + 1}][{i + 1}/{len(train_loader)}]  '
                   f'Lr {cur_lr:.6f}  '
                   f'[inte {inte_loss:.5f} + grad {grad_loss:.4f} + '
                   f'msssim {msssim_loss:.4f} + L2 {l2_loss:.4f}]  '
                   f'PSNR {psnr:.2f}')
            logger.info(msg)


if __name__ == '__main__':
    main()
