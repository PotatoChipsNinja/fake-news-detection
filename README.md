# Fake News Detection
## Requirements
- Python >= 3.9.7
- PyTorch >= 1.8.1
- scikit-learn >= 1.0.2
- tqdm >= 4.62.3
- Transformers >= 4.15.0

## Datasets
- weibo21

## Models
- BERT
- TextCNN
- BiGRU
- EANN
- MDFEND

## Usage
**For CPU:**
``` shell
$ python main.py --model textcnn  # train
$ python main.py --model textcnn --test ./params/params_textcnn.pt  # test
```
**For GPU:**
``` shell
$ python main.py --cuda --gpu 0 --model textcnn  # train
$ python main.py --cuda --gpu 0 --model textcnn --test ./params/params_textcnn.pt  # test
```

| Argument | Type | Description | Default |
|:-:|:-:|:-:|:-:|
| `--cuda` | Boolean | Use CUDA | `False` |
| `--gpu` | String | `CUDA_VISIBLE_DEVICES` | `None` |
| `--model` | String | Model name | `None` |
| `--data-dir` | String | The directory of datasets | `./data` |
| `--pretrain-model` | String | Choose the pretrained model to use | `bert-base-chinese` |
| `--pretrain-dim` | Integer | The dimension of features output by pretrained model | `768` |
| `--pretrain-dir` | String | Directory of `transformers` cache | `./pretrain` |
| `--batch-size` | Integer | Batch size | `32` |
| `--max-len` | Integer | The maximum length of tokens | `512` |
| `--epoch` | Integer | Training epoch | `20` |
| `--lr` | Float | Learning Rate | `1e-3` |
| `--early-stop` | Integer | Early stopping epoch | `3` |
| `--dataloader-cache` | String | The path of dataloader cache file | `./data/dataloader.pkl` |
| `--model-save-dir` | String | The directory to save model parameters | `./params` |
| `--test` | String | The path of model parameters file and run test | `None` |

## Results
**Total**
| Model | Accuracy | F1 | Precision | Recall | AUC |
|:-:|:-:|:-:|:-:|:-:|:-:|
| BERT | 0.8898 | 0.8895 | 0.8899 | 0.8896 | 0.9529 |
| TextCNN | 0.8870 | 0.8866 | 0.8885 | 0.8867 | 0.9578 |
| BiGRU | 0.8871 | 0.8869 | 0.8874 | 0.8879 | 0.9527 |
| EANN | 0.8764 | 0.8761 | 0.8777 | 0.8766 | 0.9452 |
| MDFEND | **0.8958** | **0.8956** | **0.8961** | **0.8960** | **0.9598** |

**Each Domain (F1-score)**
| Model | Science | Military | Education | Accidents | Politics | Health | Finance | Entertainment | Society |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| BERT | 0.8171 | 0.9098 | **0.8715** | 0.8376 | 0.8513 | **0.9330** | **0.8821** | 0.8646 | 0.8777 |
| TextCNN | **0.8192** | 0.9201 | 0.8652 | **0.8676** | 0.8446 | 0.9153 | 0.8671 | 0.8619 | 0.8792 |
| BiGRU | 0.7881 | 0.9038 | 0.8665 | 0.8612 | **0.8651** | 0.9255 | 0.8629 | 0.8592 | 0.8759 |
| EANN | 0.8002 | **0.9302** | 0.8169 | 0.8460 | 0.8449 | 0.9149 | 0.8494 | 0.8526 | 0.8685 |
| MDFEND | 0.8025 | 0.9257 | 0.8695 | 0.8661 | 0.8613 | 0.9310 | 0.8758 | **0.8712** | **0.8897** |
