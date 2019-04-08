'''
Implementation of a simple recurrent language model for the Penn TreeBank data.

citation:
Zaremba, W., Sutskever, I., and Vinyals, O. 'Recurrent neural network regularization.'
arXiv preprint arXiv:1409.2329, 2014.
'''
import torch
import torch.nn as nn
from torch.autograd import Variable


class PTB_language_model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, n_layers, dropout, rnn_type, context_dim=None):
        super(PTB_language_model, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.word_embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.n_hidden = hidden_dim

        if rnn_type == 'LSTM':
            self.recurrent = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers)
        elif rnn_type == 'mem_cell':
            self.recurrent = memory_cell(embedding_dim, hidden_dim)
        elif rnn_type == 'SCRN':
            self.recurrent = SCRN(embedding_dim, context_dim, hidden_dim)
            self.n_context = context_dim
            self.n_hidden = context_dim + hidden_dim

        self.decoder = nn.Linear(self.n_hidden, vocab_size)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.word_embed.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        if self.rnn_type == 'LSTM':
            return (Variable(torch.zeros(self.n_layers, batch_size, self.n_hidden)).cuda(),
                    Variable(torch.zeros(self.n_layers, batch_size, self.n_hidden)).cuda())
        elif self.rnn_type == 'SCRN':
            return Variable(torch.zeros(self.n_layers, batch_size, self.n_hidden)).cuda()

    def forward(self, sequence, hidden):
        embedded = self.dropout(self.word_embed(sequence))

        if self.rnn_type == 'mem_cell':
            output, hidden, _ = self.recurrent(embedded, hidden)
        elif self.rnn_type == 'LSTM':
            output, hidden = self.recurrent(embedded, hidden)
        elif self.rnn_type == 'SCRN':
            output = self.recurrent(embedded, hidden)
            hidden = output

        output = self.dropout(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
