import os
import torch
from torch import nn, optim
from transformers import BertModel
from tqdm import tqdm

from functions import MLP, CNNExtractor, MaskAttention
from utils import Averager, Recorder, split, metrics

class Model(nn.Module):
    def __init__(self, domain_num, emb_dim=768, feature_kernel={1: 64, 2: 64, 3: 64, 5: 64, 10: 64}, hidden_dims=[512], dropout=0.2, expert_num=5, domain_emb_dim=768):
        super(Model, self).__init__()
        self.expert_num = expert_num
        self.attention = MaskAttention(emb_dim)
        self.embedding = nn.Embedding(domain_num, domain_emb_dim)
        self.gate = MLP(emb_dim + domain_emb_dim, hidden_dims[-1:], expert_num, 0)
        self.expert = nn.ModuleList([CNNExtractor(feature_kernel, emb_dim) for i in range(expert_num)])
        mlp_input_shape = sum([feature_num for _, feature_num in feature_kernel.items()])
        self.classifier = MLP(mlp_input_shape, hidden_dims, 1, dropout)

    def forward(self, feature, mask, category):
        attention_feature, _ = self.attention(feature, mask)
        domain_emb = self.embedding(category)

        gate_input = torch.cat((attention_feature, domain_emb), dim=1)
        gate_output = self.gate(gate_input)
        gate_output = torch.softmax(gate_output, dim=1)

        shared_feature = sum([self.expert[i](feature) * gate_output[:, i].unsqueeze(1) for i in range(self.expert_num)])
        output = self.classifier(shared_feature)
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
        self.model_save_path = os.path.join(model_save_dir, 'params_mdfend.pkl')
        self.model = Model(len(category_dict), emb_dim=pretrain_dim).to(device)
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
                label = batch['label'].float().to(self.device)
                category = batch['category'].to(self.device)

                self.optimizer.zero_grad()
                output = self.model(feature, attention_mask, category)
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
            with torch.no_grad():
                output = self.model(feature, attention_mask, batch['category'].to(self.device))
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
