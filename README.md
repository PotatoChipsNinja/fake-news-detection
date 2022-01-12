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
| Model | Accuracy | F1 | Precision | Recall | AUC |
|:-:|:-:|:-:|:-:|:-:|:-:|
| BERT | 0.8305 | 0.8305 | 0.8330 | 0.8326 | 0.9130 |
| TextCNN | 0.8981 | 0.8981 | 0.8993 | 0.8998 | 0.9590 |
| EANN | 0.8731 | 0.8730 | 0.8728 | 0.8735 | 0.9448 |
| BiGRU | 0.8877 | 0.8876 | 0.8878 | 0.8887 | 0.9524 |
| MDFEND | 0.9017 | 0.9017 | 0.9050 | 0.9042 | 0.9613 |
