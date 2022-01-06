import os
import sys
import torch
import argparse

from dataloader import get_dataloader
from model import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--pretrain-dir', default='./pretrain')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-len', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dataloader-cache', default='./cache/dataloader.pkl')
    parser.add_argument('--model-cache', default='./cache/model.pkl')
    return parser.parse_args()

def main(args):
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    category_dict = {
        "科技": 0,
        "军事": 1,
        "教育考试": 2,
        "灾难事故": 3,
        "政治": 4,
        "医药健康": 5,
        "财经商业": 6,
        "文体娱乐": 7,
        "社会生活": 8,
        "无法确定": 9
    }

    train_dataloader, val_dataloader, test_dataloader = get_dataloader(args.data_dir, args.pretrain_dir, args.batch_size, category_dict, args.max_len, args.dataloader_cache)
    trainer = Trainer(device, args.pretrain_dir, train_dataloader, val_dataloader, test_dataloader, args.epoch, args.lr, args.model_cache)
    trainer.train()

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
