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
| `--seed` | Integer | random seed | `2022` |
| `--model` | String | Model name | `None` |
| `--data-dir` | String | The directory of datasets | `./data` |
| `--pretrain-model` | String | Choose the pretrained model to use | `bert-base-chinese` |
| `--pretrain-dim` | Integer | The dimension of features output by pretrained model | `768` |
| `--pretrain-dir` | String | Directory of `transformers` cache | `./pretrain` |
| `--batch-size` | Integer | Batch size | `64` |
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
| BERT | 0.8918 | 0.8904 | 0.8909 | 0.8902 | 0.9547 |
| BERTExt | 0.9073 | 0.9063 | 0.9061 | 0.9068 | 0.9658 |
| TextCNN | 0.8971 | 0.8960 | 0.8955 | 0.8967 | 0.9635 |
| TextCNNExt | 0.9185 | 0.9176 | 0.9174 | 0.9178 | 0.9748 |
| BiGRU | 0.8918 | 0.8905 | 0.8906 | 0.8906 | 0.9583 |
| BiGRUExt | 0.9048 | 0.9034 | 0.9044 | 0.9027 | 0.9650 |
| EANN | 0.8776 | 0.8759 | 0.8769 | 0.8758 | 0.9486 |
| EANNExt | 0.9046 | 0.9035 | 0.9033 | 0.9042 | 0.9638 |
| MDFEND | 0.9075 | 0.9066 | 0.9056 | 0.9081 | 0.9686 |
| MDFENDExt | 0.9236 | 0.9227 | 0.9222 | 0.9234 | 0.9754 |

**Each Domain (F1-score)**
| Model | Science | Military | Education | Accidents | Politics | Health | Finance | Entertainment | Society |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| BERT | 0.8641 | 0.9015 | 0.8686 | 0.8289 | 0.8641 | 0.9274 | 0.8763 | 0.8687 | 0.8792 |
| BERTExt | 0.9096 | 0.9568 | 0.8845 | 0.8471 | 0.8709 | 0.9116 | 0.884 | 0.8926 | 0.9069 |
| TextCNN | 0.8229 | 0.9323 | 0.8605 | 0.8816 | 0.8617 | 0.9317 | 0.8835 | 0.8641 | 0.8840 |
| TextCNNExt | 0.8876 | 0.9630 | 0.8998 | 0.8777 | 0.8689 | 0.9349 | 0.8938 | 0.9108 | 0.9145 |
| BiGRU | 0.8375 | 0.9015 | 0.8796 | 0.8560 | 0.8700 | 0.9178 | 0.8869 | 0.8634 | 0.8743 |
| BiGRUExt | 0.9032 | 0.9506 | 0.8792 | 0.8649 | 0.8760 | 0.9275 | 0.8763 | 0.8919 | 0.8923 |
| EANN | 0.8115 | 0.9012 | 0.8563 | 0.8217 | 0.8469 | 0.9099 | 0.8624 | 0.8626 | 0.8637 |
| EANNExt | 0.8732 | 0.9568 | 0.8760 | 0.8423 | 0.8671 | 0.9189 | 0.8691 | 0.8926 | 0.9092 |
| MDFEND | 0.8443 | 0.9354 | 0.8891 | 0.8752 | 0.8840 | 0.9468 | 0.8904 | 0.8845 | 0.8897 |
| MDFENDExt | 0.874 | 0.9384 | 0.9173 | 0.8996 | 0.8964 | 0.9446 | 0.9145 | 0.913 | 0.9085 |
