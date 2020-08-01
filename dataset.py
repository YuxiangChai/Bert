import glob
import torch.utils.data as data
from transformers import BertTokenizer
import torch


class DataSet(data.Dataset):
    def __init__(self, path):
        super(DataSet, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.sentences = []     # raw sentence
        self.labels = []        # labels
        self.tokens = []        # tokenized ids
        self.attention_masks = []   # attention_mask
        pos = glob.glob(path + 'pos/*')
        neg = glob.glob(path + 'neg/*')
        for fp in pos:
            f = open(fp, 'r', encoding='UTF-8')
            sentence = f.readline()
            encoding = self.tokenizer.encode_plus(
                sentence,
                max_length=150,
                truncation=True,
                add_special_tokens=True,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            self.tokens.append(encoding['input_ids'][0])
            self.attention_masks.append(encoding['attention_mask'][0])
            self.sentences.append(sentence)
            self.labels.append(1)

        for fp in neg:
            f = open(fp, 'r', encoding='UTF-8')
            sentence = f.readline()
            encoding = self.tokenizer.encode_plus(
                sentence,
                max_length=150,
                truncation=True,
                add_special_tokens=True,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            self.tokens.append(encoding['input_ids'][0])
            self.attention_masks.append(encoding['attention_mask'][0])
            self.sentences.append(sentence)
            self.labels.append(0)

        self.labels = torch.tensor(self.labels)

    def __getitem__(self, index):
        return self.tokens[index], self.attention_masks[index], self.labels[index]

    def __len__(self):
        return len(self.sentences)


# ds = DataSet('./aclImdb/train/')
# print(ds[0])
