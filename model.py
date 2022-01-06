import os
import torch
from torch import nn, optim
from transformers import BertModel
from tqdm import tqdm

from utils import Averager, Recorder, metrics

class BERTModel(nn.Module):
    def __init__(self, pretrain_dir):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese', cache_dir=pretrain_dir)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, input_ids, attention_mask):
        feature = self.bert(input_ids, attention_mask).last_hidden_state[:, 0]
        output = self.linear_relu_stack(feature)
        output = torch.sigmoid(output)
        return output

class Trainer:
    def __init__(self, device, pretrain_dir, train_dataloader, val_dataloader, test_dataloader, epoch, lr, model_cache):
        self.device = device
        self.epoch = epoch
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.model_cache = model_cache
        self.model = BERTModel(pretrain_dir).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        recorder = Recorder()
        if not os.path.isdir(os.path.dirname(self.model_cache)):
            os.makedirs(os.path.dirname(self.model_cache))

        for epoch in range(self.epoch):
            print('----epoch %d----' % epoch)
            self.model.train()
            avg_loss = Averager()
            for i, batch in enumerate(tqdm(self.train_dataloader)):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                label = batch['label'].float().to(self.device)
                self.optimizer.zero_grad()
                output = self.model(input_ids, attention_mask)
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
                torch.save(self.model.state_dict(), self.model_cache)
            elif decision == 'stop':
                break
            elif decision == 'continue':
                continue
        
        # load best model
        self.model.load_state_dict(torch.load(self.model_cache))
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
            with torch.no_grad():
                output = self.model(input_ids, attention_mask)
            output = output.squeeze().cpu()
            y_score = torch.cat((y_score, output))
            label = batch['label']
            y_true = torch.cat((y_true, label))
        return metrics(y_true, y_score)
