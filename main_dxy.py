#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pickle
import pdb


class CorpusDataset(Dataset):
    def __init__(self, source_data, target_data, source_word_2_idx, target_word_2_idx):
        self.source_data = source_data
        self.target_data = target_data
        self.source_word_2_idx = source_word_2_idx
        self.target_word_2_idx = target_word_2_idx

    def __getitem__(self, index):
        # create one-hot index for every word
        src = self.source_data[index]
        tgt = self.target_data[index]

        src_index = [self.source_word_2_idx[i] for i in src]
        tgt_index = [self.target_word_2_idx[i] for i in tgt]

        return src_index, tgt_index


    def batch_data_alignment(self, batch_datas):
        global device
        src_index , tgt_index = [], []
        src_len , tgt_len = [], []

        for src, tgt in batch_datas:
            src_index.append(src)
            tgt_index.append(tgt)
            src_len.append(len(src))
            tgt_len.append(len(tgt))

        max_src_len = max(src_len)
        max_tgt_len = max(tgt_len)
        src_index = [[self.source_word_2_idx["<BOS>"]] + tmp_src_index + [self.source_word_2_idx["<EOS>"]] + 
                     [self.source_word_2_idx["<PAD>"]] * (max_src_len - len(tmp_src_index)) for tmp_src_index in src_index]
        tgt_index = [[self.target_word_2_idx["<BOS>"]] + tmp_src_index + [self.target_word_2_idx["<EOS>"]] + 
                     [self.target_word_2_idx["<PAD>"]] * (max_tgt_len - len(tmp_src_index)) for tmp_src_index in tgt_index]
        src_index = torch.tensor(src_index, device=device)
        tgt_index = torch.tensor(tgt_index, device=device)

        return src_index, tgt_index


    def __len__(self):
        assert len(self.source_data) == len(self.target_data)
        return len(self.target_data)


class Encoder(nn.Module):
    def __init__(self, dim_encoder_embbeding, dim_encoder_hidden, source_corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(source_corpus_len, dim_encoder_embbeding)
        self.lstm = nn.LSTM(dim_encoder_embbeding, dim_encoder_hidden, batch_first=True)

    def forward(self, src_index):
        en_embedding = self.embedding(src_index)
        _, encoder_hidden = self.lstm(en_embedding)

        return encoder_hidden



class Decoder(nn.Module):
    def __init__(self, dim_decoder_embedding, dim_decoder_hidden, target_corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(target_corpus_len, dim_decoder_embedding)
        self.lstm = nn.LSTM(dim_decoder_embedding, dim_decoder_hidden, batch_first=True)

    def forward(self, decoder_input, hidden):
        embedding = self.embedding(decoder_input)
        decoder_output, decoder_hidden = self.lstm(embedding, hidden)

        return decoder_output, decoder_hidden
    


class Seq2Seq(nn.Module):
    def __init__(self, dim_encoder_embbeding, dim_encoder_hidden, source_corpus_len, 
                 dim_decoder_embedding, dim_decoder_hidden, target_corpus_len):
        super().__init__()
        self.encoder = Encoder(dim_encoder_embbeding, dim_encoder_hidden, source_corpus_len)
        self.decoder = Decoder(dim_decoder_embedding, dim_decoder_hidden, target_corpus_len)
        self.classifier = nn.Linear(dim_decoder_hidden, target_corpus_len)

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, src_index, tgt_index):
        decoder_input = tgt_index[:, :-1]
        label = tgt_index[:, 1:]

        encoder_hidden = self.encoder(src_index)
        decoder_output, _ = self.decoder(decoder_input, encoder_hidden)

        pre = self.classifier(decoder_output)
        loss = self.ce_loss(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))

        return loss


def generate_sentence(sentence):
    global source_word_2_idx, model, device, target_word_2_idx, target_idx_2_word

    src_index = torch.tensor([[source_word_2_idx[i] for i in sentence]], device=device)

    result = []
    encoder_hidden = model.encoder(src_index)
    decoder_input = torch.tensor([[target_word_2_idx["<BOS>"]]], device=device)

    decoder_hidden = encoder_hidden
    while True:
        decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
        pre = model.classifier(decoder_output)

        w_index = int(torch.argmax(pre, dim=-1))
        word = target_idx_2_word[w_index]

        if word == "<EOS>" or len(result) > 40:
            break

        result.append(word)
        decoder_input = torch.tensor([[w_index]], device=device)

    return "".join(result)


if __name__ == '__main__':
    
    ### configs
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    num_corpus = 300
    num_test_corpus = 10
    txt_file_path = "./天龙八部.txt"
    num_epochs = 50
    lr = 0.001

    dim_encoder_embbeding = 150
    dim_encoder_hidden = 100
    dim_decoder_embedding = 150
    dim_decoder_hidden = 100


    ### data preparation

    char_to_be_replaced = "\n 0123456789qwertyuiopasdfghjklzxcvbnm[]{};':\",./<>?ａｎｔｉ－ｃｌｉｍａｘ＋．／０１２３４５６７８９＜＝＞＠Ａ" \
                          "ＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＶＷＸＹＺ［＼］ｂｄｅｆｇｈｊｋｏｐｒｓ" \
                          "ｕｖｗｙｚ￣\u3000\x1a"

    source_target_corpus_ori = []

    with open(txt_file_path, "r", encoding="gbk", errors="ignore") as tmp_file:
        tmp_file_context = tmp_file.read()
        for tmp_char in char_to_be_replaced:
            tmp_file_context = tmp_file_context.replace(tmp_char, "")
        tmp_file_context = tmp_file_context.replace("本书来自免费小说下载站更多更新免费电子书请关注", "")

        tmp_file_sentences = tmp_file_context.split("。")
        for tmp_idx, tmp_sentence in enumerate(tmp_file_sentences):
            if ("段" in tmp_sentence) and (10 <= len(tmp_sentence) <= 40) and (10 <= len(tmp_file_sentences[tmp_idx + 1]) <= 40):
                source_target_corpus_ori.append((tmp_file_sentences[tmp_idx], tmp_file_sentences[tmp_idx + 1]))

    sample_indexes = random.sample(list(range(len(source_target_corpus_ori))), num_corpus)
    source_corpus, target_corpus = [], []
    for idx in sample_indexes:
        source_corpus.append(source_target_corpus_ori[idx][0])
        target_corpus.append(source_target_corpus_ori[idx][1])

    test_corpus = []
    for idx in range(len(source_target_corpus_ori)):
        if idx not in sample_indexes:
            test_corpus.append((source_target_corpus_ori[idx][0], source_target_corpus_ori[idx][1]))
    test_corpus = random.sample(test_corpus, num_test_corpus)
    test_source_corpus, test_target_corpus = [], []
    for tmp_src, tmp_tgt in test_corpus:
        test_source_corpus.append(tmp_src)
        test_target_corpus.append(tmp_tgt)

    ### one-hot dict generation
    idx_cnt = 0
    word_2_idx_dict = dict()
    idx_2_word_list = list()
    for tmp_corpus in [source_corpus, target_corpus, test_source_corpus, test_target_corpus]:
        for tmp_sentence in tmp_corpus:
            for tmp_word in tmp_sentence:
                if tmp_word not in word_2_idx_dict.keys():
                    word_2_idx_dict[tmp_word] = idx_cnt
                    idx_2_word_list.append(tmp_word)
                    idx_cnt += 1

    one_hot_dict_len = len(word_2_idx_dict)
    word_2_idx_dict.update({"<PAD>": one_hot_dict_len, "<BOS>": one_hot_dict_len + 1, "<EOS>": one_hot_dict_len + 2})
    idx_2_word_list += ["<PAD>", "<BOS>", "<EOS>"]
    one_hot_dict_len += 3

    source_word_2_idx, target_word_2_idx = word_2_idx_dict, word_2_idx_dict
    source_idx_2_word, target_idx_2_word = idx_2_word_list, idx_2_word_list
    source_corpus_len, target_corpus_len = one_hot_dict_len, one_hot_dict_len


    ### dataloader
    dataset = CorpusDataset(source_corpus, target_corpus, source_word_2_idx, target_word_2_idx)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=dataset.batch_data_alignment)

    ### model
    model = Seq2Seq(dim_encoder_embbeding,
                    dim_encoder_hidden,
                    source_corpus_len,
                    dim_decoder_embedding,
                    dim_decoder_hidden,
                    target_corpus_len)
    model = model.to(device)

    ### train model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(num_epochs):
        for step, (src_index, tgt_index) in enumerate(dataloader):
            loss = model(src_index, tgt_index)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss)
        print("epoch: {}, training loss: {:.5f}".format(epoch + 1, loss))

    ### visualize training loss
    plt.figure()
    plt.plot(np.array([i+1 for i in range(num_epochs)]), losses, "b-")

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss of Seq2Seq")

    plt.savefig("./training_loss.png")


    ### generate sentences from test corpus
    for idx, (tmp_src_sentence, tmp_gt_sentence) in enumerate(test_corpus):
        tmp_generated_sentence = generate_sentence(tmp_src_sentence)
        print("----------------Result {}----------------".format(idx + 1))
        print("Source sentence: {}".format(tmp_src_sentence))
        print("True target sentence: {}".format(tmp_gt_sentence))
        print("Generated target sentence: {}".format(tmp_generated_sentence))



