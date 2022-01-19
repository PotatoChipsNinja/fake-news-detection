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
| BERT | 0.8863 | 0.8844 | 0.8846 | 0.8847 | 0.9516 |
| TextCNN | 0.8914 | 0.8898 | 0.8900 | 0.8915 | 0.9569 |
| BiGRU | 0.8788 | 0.8774 | 0.8762 | 0.8802 | 0.9488 |
| EANN | 0.8715 | 0.8696 | 0.8699 | 0.8712 | 0.9435 |
| MDFEND | 0.8937 | 0.8922 | 0.8915 | 0.8935 | 0.9594 |
| My | **0.8997** | **0.8979** | **0.8984** | **0.8982** | **0.9605** |

**Each Domain (F1-score)**
| Model | Science | Military | Education | Accidents | Politics | Health | Finance | Entertainment | Society |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| BERT | 0.8174 | 0.9014 | 0.8722 | 0.8283 | 0.8684 | **0.9317** | 0.8787 | 0.8539 | 0.8727 |
| TextCNN | 0.8135 | 0.9137 | 0.8659 | **0.8673** | **0.8704** | 0.9257 | 0.8728 | 0.8722 | 0.8754 |
| BiGRU | 0.7621 | 0.8906 | 0.8470 | 0.8472 | 0.8659 | 0.9249 | 0.8592 | 0.8525 | 0.8651 |
| EANN | 0.7852 | 0.9074 | 0.8429 | 0.8418 | 0.8502 | 0.9047 | 0.8472 | 0.8459 | 0.8595 |
| MDFEND | 0.8160 | 0.9153 | **0.8779** | 0.8554 | 0.8688 | 0.9312 | 0.8755 | 0.8646 | 0.8844 |
| My | **0.8909** | **0.9538** | 0.8590 | 0.8536 | 0.8634 | 0.9035 | **0.8840** | **0.8747** | **0.8972** |
