import os
import torch
from torch import nn, optim

from utils.functions import CNNExtractor, MLP, ReverseLayerF
from utils.trainer import Trainer as TrainerBase

class Model(nn.Module):
    def __init__(self, domain_num, emb_dim=768, feature_kernel={1: 64, 2: 64, 3: 64, 5: 64, 10: 64}, hidden_dims=[512], dropout=0.2, alpha=1, post_dim=200, user_dim=200):
        super(Model, self).__init__()
        self.alpha = alpha
        self.convs = CNNExtractor(feature_kernel, emb_dim)
        self.post_embedding = MLP(3, hidden_dims[-1:], post_dim, dropout)
        self.user_embedding = MLP(3, hidden_dims[-1:], user_dim, dropout)
        mlp_input_shape = sum([feature_num for _, feature_num in feature_kernel.items()]) + post_dim + user_dim
        self.label_classifier = MLP(mlp_input_shape, hidden_dims, 1, dropout)
        self.domain_classifier = MLP(mlp_input_shape, hidden_dims, domain_num, dropout)

    def forward(self, feature, post_info, user_info):
        feature = self.convs(feature)
        post_emb = self.post_embedding(post_info)
        user_emb = self.user_embedding(user_info)
        mix_feature = torch.cat((feature, post_emb, user_emb), dim=1)
        reverse_feature = ReverseLayerF.apply(mix_feature, self.alpha)
        label_pred = self.label_classifier(mix_feature)
        domain_pred = self.domain_classifier(reverse_feature)
        return torch.sigmoid(label_pred), domain_pred

class Trainer(TrainerBase):
    def __init__(self, device, pretrain_model, pretrain_dim, pretrain_dir, train_dataloader, val_dataloader, test_dataloader, epoch, lr, early_stop, model_save_dir, category_dict):
        super(Trainer, self).__init__(device, pretrain_model, pretrain_dim, pretrain_dir, train_dataloader, val_dataloader, test_dataloader, epoch, lr, early_stop, model_save_dir, category_dict)
        self.model_save_path = os.path.join(model_save_dir, 'params_eannExt.pt')
        self.model = Model(len(category_dict), emb_dim=pretrain_dim).to(device)
        self.label_criterion = nn.BCELoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_loss(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        feature = self.bert(input_ids, attention_mask).last_hidden_state.detach()
        post_info = torch.cat((batch['commentsCount'].unsqueeze(1), batch['repostsCount'].unsqueeze(1), batch['praiseCount'].unsqueeze(1)), dim=1).float().to(self.device)
        user_info = torch.cat((batch['userFollowCount'].unsqueeze(1), batch['userFanCount'].unsqueeze(1), batch['userWeiboCount'].unsqueeze(1)), dim=1).float().to(self.device)
        label = batch['label'].float().to(self.device)
        category = batch['category'].to(self.device)

        label_pred, domain_pred = self.model(feature, post_info, user_info)
        label_pred = label_pred.squeeze()
        label_loss = self.label_criterion(label_pred, label)
        domain_loss = self.domain_criterion(domain_pred, category)
        loss = label_loss + domain_loss
        return loss

    def get_output(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        feature = self.bert(input_ids, attention_mask).last_hidden_state.detach()
        post_info = torch.cat((batch['commentsCount'].unsqueeze(1), batch['repostsCount'].unsqueeze(1), batch['praiseCount'].unsqueeze(1)), dim=1).float().to(self.device)
        user_info = torch.cat((batch['userFollowCount'].unsqueeze(1), batch['userFanCount'].unsqueeze(1), batch['userWeiboCount'].unsqueeze(1)), dim=1).float().to(self.device)
        with torch.no_grad():
            output, _ = self.model(feature, post_info, user_info)
        return output
