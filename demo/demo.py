"""Visualize HLA correspondences for an image pair.
Run from the repository root: python -m demo.demo --image1 ... --image2 ... --ckpt_path ...
"""
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
from loguru import logger
from demo.utils import HLA, cfg
from src.utils.dataset import read_megadepth_color
from src.utils.plotting import make_confidence_figure


def main():
    parser = argparse.ArgumentParser(description='HLA local feature matching demo')
    parser.add_argument('--image1', type=Path, required=True)
    parser.add_argument('--image2', type=Path, required=True)
    parser.add_argument('--ckpt_path', type=Path, required=True,
                        help='Full HLA Lightning checkpoint trained for this architecture')
    parser.add_argument('--output_dir', type=Path, default=Path('outputs/demo'))
    args = parser.parse_args()
    for path in (args.image1, args.image2, args.ckpt_path):
        if not path.is_file():
            parser.error(f'File does not exist: {path}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HLA(str(args.ckpt_path), cfg).eval().to(device)
    image0, _, mask0, _ = read_megadepth_color(str(args.image1), 832, 16, True)
    image1, _, mask1, _ = read_megadepth_color(str(args.image2), 832, 16, True)
    mask0 = F.interpolate(mask0[None, None].float(), scale_factor=0.125,
                          mode='nearest', recompute_scale_factor=False)[0].bool()
    mask1 = F.interpolate(mask1[None, None].float(), scale_factor=0.125,
                          mode='nearest', recompute_scale_factor=False)[0].bool()
    batch = {'imagec_0': image0.to(device), 'imagec_1': image1.to(device),
             'mask0': mask0.to(device), 'mask1': mask1.to(device)}
    with torch.inference_mode():
        model(batch)
    make_confidence_figure(batch, path=str(args.output_dir/'matches.png'),
                           dpi=200, topk=4000)
    logger.info('Saved HLA matching visualization to {}', args.output_dir/'matches.png')


if __name__ == '__main__':
    main()
