import os
import torch
from torch import nn, optim
from transformers import BertModel
from tqdm import tqdm

from functions import CNNExtractor, MLP, ReverseLayerF
from utils import Averager, Recorder, split, metrics

class Model(nn.Module):
    def __init__(self, domain_num, emb_dim=768, feature_kernel={1: 64, 2: 64, 3: 64, 5: 64, 10: 64}, hidden_dims=[512], dropout=0.2, alpha=1):
        super(Model, self).__init__()
        self.alpha = alpha
        self.convs = CNNExtractor(feature_kernel, emb_dim)
        mlp_input_shape = sum([feature_num for _, feature_num in feature_kernel.items()])
        self.label_classifier = MLP(mlp_input_shape, hidden_dims, 1, dropout)
        self.domain_classifier = MLP(mlp_input_shape, hidden_dims, domain_num, dropout)

    def forward(self, feature):
        feature = self.convs(feature)
        reverse_feature = ReverseLayerF.apply(feature, self.alpha)
        label_pred = self.label_classifier(feature)
        domain_pred = self.domain_classifier(reverse_feature)
        return torch.sigmoid(label_pred), domain_pred

class Trainer:
    def __init__(self, device, pretrain_model, pretrain_dim, pretrain_dir, train_dataloader, val_dataloader, test_dataloader, epoch, lr, early_stop, model_save_dir, category_dict):
        self.device = device
        self.epoch = epoch
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.early_stop = early_stop
        self.category_dict = category_dict
        self.model_save_path = os.path.join(model_save_dir, 'params_eann.pt')
        self.model = Model(len(category_dict), emb_dim=pretrain_dim).to(device)
        self.label_criterion = nn.BCELoss()
        self.domain_criterion = nn.CrossEntropyLoss()
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
                label_pred, domain_pred = self.model(feature)
                label_pred = label_pred.squeeze()
                label_loss = self.label_criterion(label_pred, label)
                domain_loss = self.domain_criterion(domain_pred, category)
                loss = label_loss + domain_loss
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
                output, _ = self.model(feature)
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
