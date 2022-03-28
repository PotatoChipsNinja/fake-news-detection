import os
import sys
import torch
import argparse

from dataloader import get_dataloader
from model.bert import Trainer as BERTTrainer
from model.bertExt import Trainer as BERTExtTrainer
from model.textcnn import Trainer as TextCNNTrainer
from model.textcnnExt import Trainer as TextCNNExtTrainer
from model.eann import Trainer as EANNTrainer
from model.eannExt import Trainer as EANNExtTrainer
from model.bigru import Trainer as BiGRUTrainer
from model.bigruExt import Trainer as BiGRUExtTrainer
from model.mdfend import Trainer as MDFENDTrainer
from model.mdfendExt import Trainer as MDFENDExtTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu')
    parser.add_argument('--model')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--pretrain-model', default='bert-base-chinese')
    parser.add_argument('--pretrain-dim', type=int, default=768)
    parser.add_argument('--pretrain-dir', default='./pretrain')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-len', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--early-stop', type=int, default=3)
    parser.add_argument('--dataloader-cache', default='./data/dataloader.pkl')
    parser.add_argument('--model-save-dir', default='./params')
    parser.add_argument('--test')
    return parser.parse_args()

def main(args):
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    if not os.path.isdir(args.model_save_dir):
        os.makedirs(args.model_save_dir)

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

    train_dataloader, val_dataloader, test_dataloader = get_dataloader(args.data_dir, args.pretrain_model, args.pretrain_dir, args.batch_size, category_dict, args.max_len, args.dataloader_cache)
    if args.model == 'bert':
        trainer = BERTTrainer(device, args.pretrain_model, args.pretrain_dim, args.pretrain_dir, train_dataloader, val_dataloader, test_dataloader, args.epoch, args.lr, args.early_stop, args.model_save_dir, category_dict)
    elif args.model == 'bertExt':
        trainer = BERTExtTrainer(device, args.pretrain_model, args.pretrain_dim, args.pretrain_dir, train_dataloader, val_dataloader, test_dataloader, args.epoch, args.lr, args.early_stop, args.model_save_dir, category_dict)
    elif args.model == 'textcnn':
        trainer = TextCNNTrainer(device, args.pretrain_model, args.pretrain_dim, args.pretrain_dir, train_dataloader, val_dataloader, test_dataloader, args.epoch, args.lr, args.early_stop, args.model_save_dir, category_dict)
    elif args.model == 'textcnnExt':
        trainer = TextCNNExtTrainer(device, args.pretrain_model, args.pretrain_dim, args.pretrain_dir, train_dataloader, val_dataloader, test_dataloader, args.epoch, args.lr, args.early_stop, args.model_save_dir, category_dict)
    elif args.model == 'eann':
        trainer = EANNTrainer(device, args.pretrain_model, args.pretrain_dim, args.pretrain_dir, train_dataloader, val_dataloader, test_dataloader, args.epoch, args.lr, args.early_stop, args.model_save_dir, category_dict)
    elif args.model == 'eannExt':
        trainer = EANNExtTrainer(device, args.pretrain_model, args.pretrain_dim, args.pretrain_dir, train_dataloader, val_dataloader, test_dataloader, args.epoch, args.lr, args.early_stop, args.model_save_dir, category_dict)
    elif args.model == 'bigru':
        trainer = BiGRUTrainer(device, args.pretrain_model, args.pretrain_dim, args.pretrain_dir, train_dataloader, val_dataloader, test_dataloader, args.epoch, args.lr, args.early_stop, args.model_save_dir, category_dict)
    elif args.model == 'bigruExt':
        trainer = BiGRUExtTrainer(device, args.pretrain_model, args.pretrain_dim, args.pretrain_dir, train_dataloader, val_dataloader, test_dataloader, args.epoch, args.lr, args.early_stop, args.model_save_dir, category_dict)
    elif args.model == 'mdfend':
        trainer = MDFENDTrainer(device, args.pretrain_model, args.pretrain_dim, args.pretrain_dir, train_dataloader, val_dataloader, test_dataloader, args.epoch, args.lr, args.early_stop, args.model_save_dir, category_dict)
    elif args.model == 'mdfendExt':
        trainer = MDFENDExtTrainer(device, args.pretrain_model, args.pretrain_dim, args.pretrain_dir, train_dataloader, val_dataloader, test_dataloader, args.epoch, args.lr, args.early_stop, args.model_save_dir, category_dict)
    else:
        print('There is no model called "%s"' % args.model)
        return -1

    if args.test is None:
        trainer.train()
    else:
        trainer.model.load_state_dict(torch.load(args.test))
        results = trainer.test(test_dataloader)
        print(results)

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
