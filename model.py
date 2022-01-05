import torch
from torch import nn, optim
from transformers import BertModel
from tqdm import tqdm

from utils import Averager

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
        return output

class Trainer:
    def __init__(self, device, pretrain_dir, train_dataloader, val_dataloader, test_dataloader, epoch, lr):
        self.device = device
        self.epoch = epoch
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.model = BERTModel(pretrain_dir).to(device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        for epoch in range(self.epoch):
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
            print('epoch %d: loss = %.4f' % (epoch+1, avg_loss.get()))

    def test(self, dataloader):
        pass
