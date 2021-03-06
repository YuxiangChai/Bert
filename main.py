import torch
import argparse
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
from tqdm import tqdm

from model import Bert
from dataset import DataSet
from transformers import AdamW, get_linear_schedule_with_warmup

EPOCH = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, loader, criterion, optimizer, scheduler):
    model.to(device)
    model.train()
    epoch_loss = 0
    for i, data in enumerate(tqdm(loader, desc='training.....')):
        input_ids = data[0].to(device)
        attention_mask = data[1].to(device)
        label = data[2].to(device)
        optimizer.zero_grad()
        output = model(input_ids, attention_mask)
        _, pred = torch.max(output, dim=1)
        loss = criterion(output, label)
        epoch_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    tqdm.write('Loss: {}'.format(epoch_loss))


def val(model, loader):
    model.to(device)
    model.eval()
    correct = 0
    wrong = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc='validating...')):
            input_ids = data[0].to(device)
            attention_mask = data[1].to(device)
            label = data[2].to(device)

            output = model(input_ids, attention_mask)
            _, pred = torch.max(output, dim=1)
            for j in range(len(pred)):
                if pred[j] == label[j]:
                    correct += 1
                else:
                    wrong += 1

    tqdm.write('Accuracy: {:.2f}'.format(correct / (correct + wrong)))


def test(model, loader):
    model.to(device)
    model.eval()
    correct = 0
    wrong = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc='testing......')):
            input_ids = data[0].to(device)
            attention_mask = data[1].to(device)
            label = data[2].to(device)

            output = model(input_ids, attention_mask)
            _, pred = torch.max(output, dim=1)
            for j in range(len(pred)):
                if pred[j] == label[j]:
                    correct += 1
                else:
                    wrong += 1

    tqdm.write('Accuracy: {:.2f}'.format(correct / (correct + wrong)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='./aclImdb/train/')
    parser.add_argument('--test', type=str, default='./aclImdb/test/')
    args = parser.parse_args()

    # create data loader for training and validation
    ds = DataSet(args.train)
    train_size = int(0.9*len(ds))
    val_size = len(ds) - train_size
    train_set, val_set = random_split(ds, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=20, shuffle=False)

    print('Train data Loaded.')

    model = Bert(2)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_loader) * EPOCH
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        tqdm.write('Epoch: {}'.format(epoch+1))
        train(model, train_loader, criterion, optimizer, scheduler)
        val(model, val_loader)

    tqdm.write('\nFinal test...')
    test_set = DataSet(args.test)
    test_loader = DataLoader(test_set, batch_size=20)
    test(model, test_loader)


if __name__ == '__main__':
    main()
