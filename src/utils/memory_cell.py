'''
Implementation of the memory cell recurrent layer.

citation:
Levy, Omer, et al. "Long Short-Term Memory as a Dynamically Computed Element-wise Weighted Sum."
Proceedings of the 56th ACL (Volume 2: Short Papers). Vol. 2. 2018.
'''
import math
import torch
import torch.nn as nn

class memory_cell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(memory_cell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.i2h = nn.Linear(input_dim, 3 * hidden_dim)  # these will be used to compute the gate activations
        self.h2h = nn.Linear(hidden_dim, 3 * hidden_dim)

        self.i2c = nn.Linear(input_dim, hidden_dim)  # this will provide the input to the cell state

        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward_step(self, x, hidden):
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        lin = self.i2h(x) + self.h2h(h)  # gates are the same, depend on both current obs and previous hidden state
        i_gate, f_gate, o_gate = lin.chunk(3, 1)

        content = self.i2c(x)  # content layer is now just linear function of input - no recurrence!

        i_gate = i_gate.sigmoid()
        f_gate = f_gate.sigmoid()
        o_gate = o_gate.sigmoid()

        c_new = (i_gate * content) + (f_gate * c)  # compute cell and hidden states
        h_new = o_gate * c_new.tanh()

        h_new = h_new.view(1, h_new.size(0), -1)
        c_new = c_new.view(1, c_new.size(0), -1)
        return h_new, (h_new, c_new), (i_gate, f_gate, o_gate), content

    def forward_with_hidden(self, seq, hidden_type='noise'):
        assert (hidden_type in ['noise', 'zero']), 'hidden_type must be either \'noise\' or \'zero\''
        seq_len = seq.size(0)

        if hidden_type == 'zero':
            h, c = (Variable(torch.zeros(1, 1, self.hidden_dim)),
                    Variable(torch.zeros(1, 1, self.hidden_dim)))
        else:
            h, c = (Variable(torch.randn(1, 1, self.hidden_dim)),
                    Variable(torch.randn(1, 1, self.hidden_dim)))

        for i in range(seq_len):
            h_new, c_new, gates, content = self.forward_step(seq[i, :, :].unsqueeze(0), (h, c))
            h_out = h_new if i == 0 else torch.cat(
                [h_out, h_new])  # record hidden states, content, gates for inspection
            i_gates = gates[0] if i == 0 else torch.cat([i_gates, gates[0]])
            f_gates = gates[1] if i == 0 else torch.cat([f_gates, gates[1]])
            cont = content if i == 0 else torch.cat([cont, content])
            if hidden_type == 'zero':
                h, c = (Variable(torch.zeros(1, 1, self.hidden_dim)),
                        Variable(torch.zeros(1, 1, self.hidden_dim)))
            else:
                h, c = (Variable(torch.randn(1, 1, self.hidden_dim)),
                        Variable(torch.randn(1, 1, self.hidden_dim)))

        return h_out, c_new, (i_gates, f_gates, cont)

    def forward(self, seq, init_hidd):  # call forward_step() iteratively over sequence
        seq_len = seq.size(0)
        h, c = init_hidd
        for i in range(seq_len):
            h_new, c_new, gates, content = self.forward_step(seq[i, :, :].unsqueeze(0), (h, c))
            h_out = h_new if i == 0 else torch.cat(
                [h_out, h_new])  # record hidden states, content, gates for inspection
            i_gates = gates[0] if i == 0 else torch.cat([i_gates, gates[0]])
            f_gates = gates[1] if i == 0 else torch.cat([f_gates, gates[1]])
            cont = content if i == 0 else torch.cat([cont, content])
            h, c = c_new
        return h_out, c_new, (i_gates, f_gates, cont)
