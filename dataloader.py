import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class Weibo21Dataset(Dataset):
    def __init__(self, path, category_dict, pretrain_model, pretrain_dir, max_len):
        self.category_dict = category_dict
        self.max_len = max_len
        try:
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_dir)
        except:
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_model, cache_dir=os.path.join(pretrain_dir, 'cache'))
            self.tokenizer.save_pretrained(pretrain_dir)
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        inputs = self.tokenizer(item['content'], max_length=self.max_len, padding='max_length', truncation=True)
        return {
            'input_ids': torch.tensor(inputs['input_ids']),
            'attention_mask': torch.tensor(inputs['attention_mask']),
            'category': self.category_dict[item['category']],
            'label': item['label'],
            'commentsCount': item['commentsCount'],
            'repostsCount': item['repostsCount'],
            'praiseCount': item['praiseCount'],
            'userFollowCount': item['userFollowCount'],
            'userFanCount': item['userFanCount'],
            'userWeiboCount': item['userWeiboCount']
        }

def get_dataloader(data_dir, pretrain_model, pretrain_dir, batch_size, category_dict, max_len, dataloader_cache):
    if os.path.isfile(dataloader_cache):
        with open(dataloader_cache, 'rb') as f:
            train_dataloader, val_dataloader, test_dataloader = pickle.load(f)
    else:
        train_path = os.path.join(data_dir, 'train_only_9_domain.pkl')
        val_path = os.path.join(data_dir, 'val_only_9_domain.pkl')
        test_path = os.path.join(data_dir, 'test_only_9_domain.pkl')

        train_dataset = Weibo21Dataset(train_path, category_dict, pretrain_model, pretrain_dir, max_len)
        val_dataset = Weibo21Dataset(val_path, category_dict, pretrain_model, pretrain_dir, max_len)
        test_dataset = Weibo21Dataset(test_path, category_dict, pretrain_model, pretrain_dir, max_len)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if not os.path.isdir(os.path.dirname(dataloader_cache)):
            os.makedirs(os.path.dirname(dataloader_cache))
        with open(dataloader_cache, 'wb') as f:
            pickle.dump((train_dataloader, val_dataloader, test_dataloader), f)

    return train_dataloader, val_dataloader, test_dataloader
