from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

class Averager:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def add(self, value):
        self.sum += value
        self.count += 1

    def get(self):
        return self.sum / self.count

class Recorder:
    def __init__(self, early_stop):
        self.early_stop = early_stop
        self.best_score = 0
        self.best_epoch = 0
        self.epoch = 0

    def update(self, score):
        self.epoch += 1
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = self.epoch
            return 'save'
        if self.epoch - self.best_epoch >= self.early_stop:
            return 'stop'
        else:
            return 'continue'

def metrics(y_true, y_score):
    results = dict()
    y_pred = y_score.round()
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['f1'] = f1_score(y_true, y_pred, average='macro')
    results['precision'] = precision_score(y_true, y_pred, average='macro')
    results['recall'] = recall_score(y_true, y_pred, average='macro')
    results['auc'] = roc_auc_score(y_true, y_score)
    return results

def split(y_true, y_score, category, category_num):
    y_per_category = []
    for category_id in range(category_num):
        y_per_category.append((y_true[category == category_id], y_score[category == category_id]))
    return y_per_category
