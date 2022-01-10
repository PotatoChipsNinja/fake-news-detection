import os
import torch
from torch import nn, optim
from torch.nn.modules.dropout import Dropout
from transformers import BertModel
from tqdm import tqdm

from utils import Averager, Recorder, metrics

class BERTModel(nn.Module):
    def __init__(self, hidden_dim=512, dropout=0.2):
        super(BERTModel, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feature):
        output = self.linear_relu_stack(feature)
        output = torch.sigmoid(output)
        return output

class Trainer:
    def __init__(self, device, pretrain_dir, train_dataloader, val_dataloader, test_dataloader, epoch, lr, early_stop, model_save_dir):
        self.device = device
        self.epoch = epoch
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.early_stop = early_stop
        self.model_save_path = os.path.join(model_save_dir, 'params_mlp.pkl')
        self.model = BERTModel().to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.bert = BertModel.from_pretrained('bert-base-chinese', cache_dir=pretrain_dir).to(device)

    def train(self):
        recorder = Recorder(self.early_stop)
        for epoch in range(self.epoch):
            print('----epoch %d----' % (epoch+1))
            self.model.train()
            avg_loss = Averager()
            for i, batch in enumerate(tqdm(self.train_dataloader)):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                feature = self.bert(input_ids, attention_mask).pooler_output.detach()
                label = batch['label'].float().to(self.device)
                self.optimizer.zero_grad()
                output = self.model(feature)
                output = output.squeeze()
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                avg_loss.add(loss.item())
            results = self.test(self.val_dataloader)
            print('epoch %d: loss = %.4f, acc = %.4f, f1 = %.4f, auc = %.4f' % (epoch+1, avg_loss.get(), results['accuracy'], results['f1'], results['auc']))

            # early stop
            decision = recorder.update(results['f1'])
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
        print('test: acc = %.4f, f1 = %.4f, auc = %.4f' % (results['accuracy'], results['f1'], results['auc']))

    def test(self, dataloader):
        self.model.eval()
        y_true = torch.empty(0)
        y_score = torch.empty(0)
        for i, batch in enumerate(tqdm(dataloader)):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            feature = self.bert(input_ids, attention_mask).pooler_output.detach()
            with torch.no_grad():
                output = self.model(feature)
            output = output.squeeze().cpu()
            y_score = torch.cat((y_score, output))
            label = batch['label']
            y_true = torch.cat((y_true, label))
        return metrics(y_true, y_score)
