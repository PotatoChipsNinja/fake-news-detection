import os
import torch
from torch import nn, optim
from transformers import BertModel
from tqdm import tqdm

from functions import MLP, MaskAttention
from utils import Averager, Recorder, split, metrics

class Model(nn.Module):
    def __init__(self, emb_dim=768, hidden_dims=[512], dropout=0.2, post_dim=200, user_dim=200):
        super(Model, self).__init__()
        self.attention = MaskAttention(emb_dim)
        self.post_embedding = MLP(3, hidden_dims[-1:], post_dim, dropout)
        self.user_embedding = MLP(3, hidden_dims[-1:], user_dim, dropout)
        mlp_input_shape = emb_dim + post_dim + user_dim
        self.mlp = MLP(mlp_input_shape, hidden_dims, 1, dropout)

    def forward(self, feature, mask, post_info, user_info):
        feature, _ = self.attention(feature, mask)
        post_emb = self.post_embedding(post_info)
        user_emb = self.user_embedding(user_info)
        mix_feature = torch.cat((feature, post_emb, user_emb), dim=1)
        output = self.mlp(mix_feature)
        output = torch.sigmoid(output)
        return output

class Trainer:
    def __init__(self, device, pretrain_model, pretrain_dim, pretrain_dir, train_dataloader, val_dataloader, test_dataloader, epoch, lr, early_stop, model_save_dir, category_dict):
        self.device = device
        self.epoch = epoch
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.early_stop = early_stop
        self.category_dict = category_dict
        self.model_save_path = os.path.join(model_save_dir, 'params_my.pt')
        self.model = Model(emb_dim=pretrain_dim).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.bert = BertModel.from_pretrained(pretrain_model, cache_dir=pretrain_dir).to(device)

    def train(self):
        recorder = Recorder(self.early_stop)
        for epoch in range(self.epoch):
            print('----epoch %d----' % (epoch+1))
            self.model.train()
            avg_loss = Averager()
            for i, batch in enumerate(tqdm(self.train_dataloader)):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                feature = self.bert(input_ids, attention_mask).last_hidden_state.detach()
                post_info = torch.cat((batch['commentsCount'].unsqueeze(1), batch['repostsCount'].unsqueeze(1), batch['praiseCount'].unsqueeze(1)), dim=1).float().to(self.device)
                user_info = torch.cat((batch['userFollowCount'].unsqueeze(1), batch['userFanCount'].unsqueeze(1), batch['userWeiboCount'].unsqueeze(1)), dim=1).float().to(self.device)
                label = batch['label'].float().to(self.device)
                self.optimizer.zero_grad()
                output = self.model(feature, attention_mask, post_info, user_info)
                output = output.squeeze()
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                avg_loss.add(loss.item())
            results = self.test(self.val_dataloader)
            print('epoch %d: loss = %.4f, acc = %.4f, f1 = %.4f, auc = %.4f' % (epoch+1, avg_loss.get(), results['total']['accuracy'], results['total']['f1'], results['total']['auc']))

            # early stop
            decision = recorder.update(results['total']['f1'])
            if decision == 'save':
                torch.save(self.model.state_dict(), self.model_save_path)
            elif decision == 'stop':
                break
            elif decision == 'continue':
                continue

        # load best model
        self.model.load_state_dict(torch.load(self.model_save_path))
        print('----test----')
        results = self.test(self.test_dataloader)
        print('test: acc = %.4f, f1 = %.4f, auc = %.4f' % (results['total']['accuracy'], results['total']['f1'], results['total']['auc']))

    def test(self, dataloader):
        self.model.eval()
        category = torch.empty(0)
        y_true = torch.empty(0)
        y_score = torch.empty(0)
        for i, batch in enumerate(tqdm(dataloader)):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            feature = self.bert(input_ids, attention_mask).last_hidden_state.detach()
            post_info = torch.cat((batch['commentsCount'].unsqueeze(1), batch['repostsCount'].unsqueeze(1), batch['praiseCount'].unsqueeze(1)), dim=1).float().to(self.device)
            user_info = torch.cat((batch['userFollowCount'].unsqueeze(1), batch['userFanCount'].unsqueeze(1), batch['userWeiboCount'].unsqueeze(1)), dim=1).float().to(self.device)
            with torch.no_grad():
                output = self.model(feature, attention_mask, post_info, user_info)
            output = output.squeeze().cpu()
            y_score = torch.cat((y_score, output))
            y_true = torch.cat((y_true, batch['label']))
            category = torch.cat((category, batch['category']))

        results = dict()
        results['total'] = metrics(y_true, y_score)
        y_per_category = split(y_true, y_score, category, len(self.category_dict))
        for category_name, category_id in self.category_dict.items():
            results[category_name] = metrics(y_per_category[category_id][0], y_per_category[category_id][1])
        return results
