import os
import torch
from torch import nn, optim

from utils.functions import CNNExtractor, MLP, ReverseLayerF
from utils.trainer import Trainer as TrainerBase

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

class Trainer(TrainerBase):
    def __init__(self, device, pretrain_model, pretrain_dim, pretrain_dir, train_dataloader, val_dataloader, test_dataloader, epoch, lr, early_stop, model_save_dir, category_dict):
        super(Trainer, self).__init__(device, pretrain_model, pretrain_dim, pretrain_dir, train_dataloader, val_dataloader, test_dataloader, epoch, lr, early_stop, model_save_dir, category_dict)
        self.model_save_path = os.path.join(model_save_dir, 'params_eann.pt')
        self.model = Model(len(category_dict), emb_dim=pretrain_dim).to(device)
        self.label_criterion = nn.BCELoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_loss(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        feature = self.bert(input_ids, attention_mask).last_hidden_state.detach()
        label = batch['label'].float().to(self.device)
        category = batch['category'].to(self.device)

        label_pred, domain_pred = self.model(feature)
        label_pred = label_pred.squeeze()
        label_loss = self.label_criterion(label_pred, label)
        domain_loss = self.domain_criterion(domain_pred, category)
        loss = label_loss + domain_loss
        return loss

    def get_output(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        feature = self.bert(input_ids, attention_mask).last_hidden_state.detach()
        with torch.no_grad():
            output, _ = self.model(feature)
        return output
