# -*- coding: utf-8 -*-
import os
import torch
from config import parse_config
from data_loader import DataBatchIterator
from data_loader import PAD
from model import TextCNN
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import logging


def build_textcnn_model(vocab, config, train=True):
    model = TextCNN(vocab.vocab_size, config)
    if train:
        model.train()
    else:
        model.eval()

    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    return model


def train_textcnn_model(model, train_data, valid_data, test_data,padding_idx, config):
    # Build optimizer.
    # params = [p for k, p in model.named_parameters(
    # ) if p.requires_grad and "embed" not in k]
    params = [p for k, p in model.named_parameters() if p.requires_grad]
    optimizer = Adam(params, lr=config.lr)
    criterion = CrossEntropyLoss(reduction="sum")

    if os.path.exists(config.save_model):
        checkpoint = torch.load(config.save_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')

    model.train()

    for epoch in range(start_epoch+1, config.epochs + 1):
        train_data_iter = iter(train_data)
        for idx, batch in enumerate(train_data_iter):
            model.zero_grad()
            ground_truth = batch.label
            # batch_first = False
            outputs = model(batch.sent)
            loss = criterion(outputs, ground_truth)
            loss.backward()
            optimizer.step()

            if idx % 20 == 0:
                valid_loss = valid_textcnn_model(
                    model, valid_data, criterion, config)
                # 处理
                print("epoch {0:d} [{1:d}/{2:d}], valid loss: {3:.2f}".format(
                    epoch, idx, train_data.num_batches, valid_loss))
                model.train()
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, config.save_model)  # 保存参数 便于下次继续训练
        acc = test(model, test_data, config)
        print("Acc:", acc)

def test(model, test_data,config):
    #checkpoint = torch.load(config.save_model)
    #model.load_state_dict(checkpoint['model'])

    model.eval()
    train_acc = 0
    total=0
    test_data_iter = iter(test_data)
    for idx, batch in enumerate(test_data_iter):
        model.zero_grad()
        ground_truth = batch.label
        # batch_first = False
        outputs = model(batch.sent)
        pred=torch.argmax(outputs, 1)
        #print(pred)
        #print(ground_truth)
        train_acc += (pred == ground_truth).sum().float()
        total+=len(ground_truth)
        break

    return train_acc/total


def valid_textcnn_model(model,  valid_data, criterion, config):
    # Build optimizer.
    # params = [p for k, p in model.named_parameters(
    # ) if p.requires_grad and "embed" not in k]
    model.eval()
    total_loss = 0
    valid_data_iter = iter(valid_data)
    for idx, batch in enumerate(valid_data_iter):
        model.zero_grad()
        ground_truth = batch.label
        # batch_first = False
        outputs = model(batch.sent)
        # probs = model.generator(decoder_outputs)
        loss = criterion(outputs, batch.label)
        # loss 打印
        # 处理
        total_loss += loss
        break
    return loss


def main():
    # 读配置文件
    config = parse_config()
    # 载入训练集合
    train_data = DataBatchIterator(
        config=config,
        is_train=True,
        dataset="train",
        batch_size=config.batch_size,
        shuffle=True)
    train_data.load()

    vocab = train_data.vocab

    # 载入测试集合
    valid_data = DataBatchIterator(
        config=config,
        is_train=False,
        dataset="dev",
        batch_size=config.batch_size)
    valid_data.set_vocab(vocab)
    valid_data.load()

    test_data = DataBatchIterator(
        config=config,
        is_train=False,
        dataset="test",
        batch_size=config.batch_size)
    test_data.set_vocab(vocab)
    test_data.load()

    # 构建textcnn模型
    model = build_textcnn_model(
        vocab, config, train=True)

    print(model)

    # Do training.
    padding_idx = vocab.stoi[PAD]
    train_textcnn_model(model, train_data,
                        valid_data, test_data,padding_idx, config)

    #torch.save(model, '%s.pt' % (config.save_model))

    # 测试时
    checkpoint = torch.load(config.save_model+".pt",
                          map_location = config.device)
    # checkpoint
    model = build_textcnn_model(
         vocab, config, train=True)
    # .....
if __name__ == "__main__":
    main()
