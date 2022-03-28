import torch
from tqdm import tqdm
from transformers import BertModel

from utils.utils import Averager, Recorder, split, metrics

class Trainer:
    def __init__(self, device, pretrain_model, pretrain_dim, pretrain_dir, train_dataloader, val_dataloader, test_dataloader, epoch, lr, early_stop, model_save_dir, category_dict):
        self.device = device
        self.epoch = epoch
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.early_stop = early_stop
        self.category_dict = category_dict
        # self.bert = BertModel.from_pretrained(pretrain_model, cache_dir=pretrain_dir).to(device)
        try:
            self.bert = BertModel.from_pretrained(pretrain_dir).to(device)
        except:
            self.bert = BertModel.from_pretrained(pretrain_model).to(device)
            self.bert.save_pretrained(pretrain_dir)
        # Need to define self.model_save_path, self.model, self.criterion and self.optimizer in derived class

    def get_loss(self, batch):
        # Need to be implemented in derived class
        return None

    def get_output(self, batch):
        # Need to be implemented in derived class
        return None

    def train(self):
        recorder = Recorder(self.early_stop)
        for epoch in range(self.epoch):
            print('----epoch %d----' % (epoch+1))
            self.model.train()
            avg_loss = Averager()
            for i, batch in enumerate(tqdm(self.train_dataloader)):
                self.optimizer.zero_grad()
                loss = self.get_loss(batch)
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
            output = self.get_output(batch)
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
