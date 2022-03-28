import os
import torch
from torch import nn, optim

from utils.functions import CNNExtractor, MLP
from utils.trainer import Trainer as TrainerBase

class Model(nn.Module):
    def __init__(self, emb_dim=768, feature_kernel={1: 64, 2: 64, 3: 64, 5: 64, 10: 64}, hidden_dims=[512], dropout=0.2, post_dim=200, user_dim=200):
        super(Model, self).__init__()
        self.convs = CNNExtractor(feature_kernel, emb_dim)
        self.post_embedding = MLP(3, hidden_dims[-1:], post_dim, dropout)
        self.user_embedding = MLP(3, hidden_dims[-1:], user_dim, dropout)
        mlp_input_shape = sum([feature_num for _, feature_num in feature_kernel.items()]) + post_dim + user_dim
        self.mlp = MLP(mlp_input_shape, hidden_dims, 1, dropout)

    def forward(self, feature, post_info, user_info):
        feature = self.convs(feature)
        post_emb = self.post_embedding(post_info)
        user_emb = self.user_embedding(user_info)
        mix_feature = torch.cat((feature, post_emb, user_emb), dim=1)
        output = self.mlp(mix_feature)
        output = torch.sigmoid(output)
        return output

class Trainer(TrainerBase):
    def __init__(self, device, pretrain_model, pretrain_dim, pretrain_dir, train_dataloader, val_dataloader, test_dataloader, epoch, lr, early_stop, model_save_dir, category_dict):
        super(Trainer, self).__init__(device, pretrain_model, pretrain_dim, pretrain_dir, train_dataloader, val_dataloader, test_dataloader, epoch, lr, early_stop, model_save_dir, category_dict)
        self.model_save_path = os.path.join(model_save_dir, 'params_textcnnExt.pt')
        self.model = Model(emb_dim=pretrain_dim).to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_loss(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        feature = self.bert(input_ids, attention_mask).last_hidden_state.detach()
        post_info = torch.cat((batch['commentsCount'].unsqueeze(1), batch['repostsCount'].unsqueeze(1), batch['praiseCount'].unsqueeze(1)), dim=1).float().to(self.device)
        user_info = torch.cat((batch['userFollowCount'].unsqueeze(1), batch['userFanCount'].unsqueeze(1), batch['userWeiboCount'].unsqueeze(1)), dim=1).float().to(self.device)
        label = batch['label'].float().to(self.device)

        output = self.model(feature, post_info, user_info)
        output = output.squeeze()
        loss = self.criterion(output, label)
        return loss

    def get_output(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        feature = self.bert(input_ids, attention_mask).last_hidden_state.detach()
        post_info = torch.cat((batch['commentsCount'].unsqueeze(1), batch['repostsCount'].unsqueeze(1), batch['praiseCount'].unsqueeze(1)), dim=1).float().to(self.device)
        user_info = torch.cat((batch['userFollowCount'].unsqueeze(1), batch['userFanCount'].unsqueeze(1), batch['userWeiboCount'].unsqueeze(1)), dim=1).float().to(self.device)
        with torch.no_grad():
            output = self.model(feature, post_info, user_info)
        return output
