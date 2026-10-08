import os
import pprint
import argparse
import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import datasets
from utils import train_util, log_util, anomaly_util
from config.defaults import _C as config, update_config
from models import Network_Myself

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='HSTA Test / Inference')
    parser.add_argument('--cfg', help='config file', default='config/avenue.yaml', type=str)
    parser.add_argument('--model-file', help='path to trained model (.pth)', required=True, type=str)
    parser.add_argument('opts', help='extra config overrides', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    update_config(config, args)

    logger, final_output_dir, _ = log_util.create_logger(config, args.cfg, 'test')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    gpus = [config.GPUS[0]]
    model = Network_Myself(config, pretrained=False)
    logger.info('Model: {}'.format(model.get_name()))
    model = nn.DataParallel(model, device_ids=gpus).cuda(device=gpus[0])

    state_dict = torch.load(args.model_file, map_location='cpu')
    if 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.module.load_state_dict(state_dict)
    logger.info('Loaded model from: {}'.format(args.model_file))

    test_dataset = datasets.get_test_data(config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    mat_loader = datasets.get_label(config)
    mat = mat_loader()

    psnr_list, anomaly_results = inference(config, test_loader, model, gpus[0])
    assert len(psnr_list) == len(mat), \
        f'Ground truth has {len(mat)} videos, but got {len(psnr_list)} detected videos!'

    auc, fpr, tpr = anomaly_util.calculate_auc(config, psnr_list, mat)

    result_path = os.path.join(final_output_dir, 'anomaly_results.csv')
    with open(result_path, 'w') as f:
        f.write("image_path,psnr,anomaly,original_video,frame_index\n")
        for res in anomaly_results:
            f.write(f"{res['image_path']},{res['psnr']:.4f},{res['anomaly']},"
                    f"{os.path.basename(res['video'])},{res['frame_in_video']}\n")

    logger.info(f'Results saved to: {result_path}')
    logger.info(f'AUC: {auc * 100:.1f}%')


def inference(config, data_loader, model, device):
    loss_func_mse = nn.MSELoss(reduction='none')
    model.eval()

    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df

    psnr_list = []
    anomaly_results = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            video, frame_path_list = train_util.decode_input(input=data, train=False)
            video = [frame.to(device=device) for frame in video]
            frame_paths = [item for sublist in frame_path_list for item in sublist]

            print(f'[{i + 1}/{len(data_loader)}]')
            psnr_video = []

            for f in tqdm.tqdm(range(len(video) - fp)):
                inputs = video[f:f + fp]
                output = model(inputs)
                target = video[f + fp]

                mse_imgs = torch.mean(loss_func_mse(
                    (output[0] + 1) / 2, (target[0] + 1) / 2
                )).item()
                psnr = anomaly_util.psnr_park(mse_imgs)
                psnr_video.append(psnr)

                is_anomaly = 1 if psnr < config.TEST.ANOMALY_THRESHOLD else 0
                anomaly_results.append({
                    'image_path': frame_paths[f + fp],
                    'psnr': psnr,
                    'anomaly': is_anomaly,
                    'video': os.path.basename(os.path.dirname(frame_paths[0])),
                    'frame_in_video': f + fp
                })

            psnr_list.append(psnr_video)

    return psnr_list, anomaly_results


if __name__ == '__main__':
    main()
