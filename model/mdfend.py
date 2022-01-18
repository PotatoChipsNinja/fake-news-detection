import os
import torch
from torch import nn, optim

from utils.functions import MLP, CNNExtractor, MaskAttention
from utils.trainer import Trainer as TrainerBase

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

class Trainer(TrainerBase):
    def __init__(self, device, pretrain_model, pretrain_dim, pretrain_dir, train_dataloader, val_dataloader, test_dataloader, epoch, lr, early_stop, model_save_dir, category_dict):
        super(Trainer, self).__init__(device, pretrain_model, pretrain_dim, pretrain_dir, train_dataloader, val_dataloader, test_dataloader, epoch, lr, early_stop, model_save_dir, category_dict)
        self.model_save_path = os.path.join(model_save_dir, 'params_mdfend.pt')
        self.model = Model(len(category_dict), emb_dim=pretrain_dim).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_loss(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        feature = self.bert(input_ids, attention_mask).last_hidden_state.detach()
        label = batch['label'].float().to(self.device)
        category = batch['category'].to(self.device)

        output = self.model(feature, attention_mask, category)
        output = output.squeeze()
        loss = self.criterion(output, label)
        return loss

    def get_output(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        feature = self.bert(input_ids, attention_mask).last_hidden_state.detach()
        category = batch['category'].to(self.device)
        with torch.no_grad():
            output = self.model(feature, attention_mask, category)
        return output
