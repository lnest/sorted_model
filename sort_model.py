# -*- coding: utf-8 -*-

# ------------------------------------
# Create On 2018/6/4 11:08 
# File Name: sort_model.py
# Edit Author: lnest
# ------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from load_data import dataloader


class Encoder(nn.Module):
    def __init__(self, word_len=2194, embeding_dim=512, hidden_size=256, drop_out=0.1):
        super(Encoder, self).__init__()
        self.embeding_dim = embeding_dim
        self.word_len = word_len
        self.hidden_size = hidden_size
        self.drop_out = nn.Dropout(drop_out)
        self.embeding = nn.Embedding(num_embeddings=word_len, embedding_dim=embeding_dim, padding_idx=0)
        self.linear = nn.Linear(embeding_dim, hidden_size)

    def forward(self, encoder_word):
        encoder = self.embeding(encoder_word)
        encoder_output = self.linear(encoder)
        encoder_output = self.drop_out(encoder_output)
        return encoder_output


class SortedModel(nn.Module):
    def __init__(self, word_len=4094, max_length=9, embeding_dim=512, hidden_size=256):
        super(SortedModel, self).__init__()
        self.max_seq_len = max_length
        self.embedding = nn.Embedding(num_embeddings=word_len, embedding_dim=embeding_dim)
        self.hidden_size = hidden_size
        self.atten = nn.Linear(self.hidden_size + self.hidden_size, max_length)
        self.atten_combine = nn.Linear(embeding_dim + self.hidden_size, embeding_dim)
        self.gru = nn.GRUCell(embeding_dim, hidden_size=self.hidden_size)
        self.output_linear = nn.Linear(hidden_size, word_len)

    def forward(self, decode_word, hidden, encoder_output):
        _embedded = self.embedding(decode_word)
        # print('decoder embeded size:', _embedded.size())
        hidden_unsqu = hidden.unsqueeze(1)
        hidden_expand = hidden_unsqu.expand(-1, self.max_seq_len, -1)
        # size: 1024 * max_len * max_len
        atten_weight = func.softmax(self.atten(torch.cat((encoder_output, hidden_expand), 2)), dim=2)
        # size: 1024 * max_len * embed_size
        atten_applied = torch.bmm(atten_weight, encoder_output)   # a(i,j) * hj
        # print('atten_applied size:', atten_applied.size())
        c_i = atten_applied.sum(1).unsqueeze(1)
        # print('c_i size', c_i.size())
        output = torch.cat((_embedded, c_i), 2)  # ci concat yi-1
        output = self.atten_combine(output)
        output = func.relu(output).squeeze(1)
        # print('output size:', output.size())
        output = self.gru(output, hidden)   # si
        hidden = output
        output = func.log_softmax(self.output_linear(output), 1)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


def train(data, epoch, encoder, decoder, criterion, print_every=3, max_len=10, learning_rate=0.01, plot_every=100):
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    running_loss = 0.0
    for i_batch, batch_data in enumerate(data, 1):
        input_data = batch_data['input']
        target_data = batch_data['output']
        decoder_hidden = decoder.init_hidden(target_data.size()[0])
        encoder_output = encoder(input_data)
        target_len = target_data.size()[1]
        decoder_input = torch.ones((encoder_output.size()[0], 1), dtype=torch.long)
        loss = 0.0
        for i in range(target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
            # print('decoder output size:', decoder_output.size())
            # print('decoder hidden size:', decoder_hidden.size())
            topv, topi = decoder_output.topk(1, dim=1)
            decoder_input = topi
            loss += criterion(decoder_output, target_data[:, i])
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        running_loss += loss.item()
        if i_batch % print_every == 0:
            print('epoch: %d, batch: %d, loss: %lf' % (epoch, i_batch, running_loss/i_batch))

    torch.save(encoder.state_dict(), './model/encoder%d.pkl' % epoch)
    torch.save(decoder.state_dict(), './model/decoder%d.pkl' % epoch)


if __name__ == '__main__':
    encoder = Encoder()
    decoder = SortedModel()
    criterion = nn.NLLLoss()
    for i in range(100):
        train(dataloader, i, encoder, decoder, criterion)
