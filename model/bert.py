import os
import torch
from torch import nn, optim

from utils.functions import MLP, MaskAttention
from utils.trainer import Trainer as TrainerBase

class Model(nn.Module):
    def __init__(self, emb_dim=768, hidden_dims=[512], dropout=0.2):
        super(Model, self).__init__()
        self.attention = MaskAttention(emb_dim)
        self.mlp = MLP(emb_dim, hidden_dims, 1, dropout)

    def forward(self, feature, mask):
        feature, _ = self.attention(feature, mask)
        output = self.mlp(feature)
        output = torch.sigmoid(output)
        return output

class Trainer(TrainerBase):
    def __init__(self, device, pretrain_model, pretrain_dim, pretrain_dir, train_dataloader, val_dataloader, test_dataloader, epoch, lr, early_stop, model_save_dir, category_dict):
        super(Trainer, self).__init__(device, pretrain_model, pretrain_dim, pretrain_dir, train_dataloader, val_dataloader, test_dataloader, epoch, lr, early_stop, model_save_dir, category_dict)
        self.model_save_path = os.path.join(model_save_dir, 'params_bert.pt')
        self.model = Model(emb_dim=pretrain_dim).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_loss(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        feature = self.bert(input_ids, attention_mask).last_hidden_state.detach()
        label = batch['label'].float().to(self.device)

        output = self.model(feature, attention_mask)
        output = output.squeeze()
        loss = self.criterion(output, label)
        return loss

    def get_output(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        feature = self.bert(input_ids, attention_mask).last_hidden_state.detach()
        with torch.no_grad():
            output = self.model(feature, attention_mask)
        return output
