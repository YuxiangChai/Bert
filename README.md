# Bert Text Classification

Use pytorch version of Bert implementation, developed by [huggingface](https://github.com/huggingface/transformers), to classify text.

## Requirements

- pytorch >= 1.5
- transformers

## Dataset

Use [Imdb review dataset](https://ai.stanford.edu/~amaas/data/sentiment/), which include 25000 for training and 25000 for testing.

I split the training set into 90% and 10% for training and validation.



## Train and Test

```
$ python main.py [--train path/] [--test path/]
```

Train path and test path are optional. The default path is './aclImdb/train/' and './aclImdb/test/'.

## Sample Result

```
Train data Loaded.
Epoch: 1
training.....: 100%|██████████| 1125/1125 [04:48<00:00,  3.90it/s]
Loss: 398.30986534804106
validating...: 100%|██████████| 125/125 [00:10<00:00, 12.18it/s]
Accuracy: 0.88

Final test...
testing......: 100%|██████████| 1250/1250 [01:42<00:00, 12.25it/s]
Accuracy: 0.89
```