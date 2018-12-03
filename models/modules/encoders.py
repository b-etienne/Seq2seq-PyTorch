import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from models.helpers import skip_add_pyramid


class EncoderRNN(nn.Module):
    def __init__(self, config):
        super(EncoderRNN, self).__init__()
        self.input_size = config["n_channels"]
        self.hidden_size = config["encoder_hidden"]
        self.layers = config.get("encoder_layers", 1)
        self.dnn_layers = config.get("encoder_dnn_layers", 0)
        self.dropout = config.get("encoder_dropout", 0.)
        self.bi = config.get("bidirectional_encoder", False)
        if self.dnn_layers > 0:
            for i in range(self.dnn_layers):
                self.add_module('dnn_' + str(i), nn.Linear(
                    in_features=self.input_size if i == 0 else self.hidden_size,
                    out_features=self.hidden_size
                ))
        gru_input_dim = self.input_size if self.dnn_layers == 0 else self.hidden_size
        self.rnn = nn.GRU(
            gru_input_dim,
            self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=True)
        self.gpu = config.get("gpu", False)

    def run_dnn(self, x):
        for i in range(self.dnn_layers):
            x = F.relu(getattr(self, 'dnn_'+str(i))(x))
        return x

    def forward(self, inputs, hidden, input_lengths):
        if self.dnn_layers > 0:
            inputs = self.run_dnn(inputs)
        x = pack_padded_sequence(inputs, input_lengths, batch_first=True)
        output, state = self.rnn(x, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True, padding_value=0.)

        if self.bi:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output, state

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(2 if self.bi else 1, batch_size, self.hidden_size))
        if self.gpu:
            h0 = h0.cuda()
        return h0


class EncoderPyRNN(nn.Module):
    def __init__(self, config):
        super(EncoderPyRNN, self).__init__()
        self.input_size = config["n_channels"]
        self.hidden_size = config["encoder_hidden"]
        self.n_layers = config.get("encoder_layers", 1)
        self.dnn_layers = config.get("encoder_dnn_layers", 0)
        self.dropout = config.get("encoder_dropout", 0.)
        self.bi = config.get("bidirectional_encoder", False)
        self.skip_add = config.get("skip_add_pyramid_encoder", "add")
        self.gpu = config.get("gpu", False)

        if self.dnn_layers > 0:
            for i in range(self.dnn_layers):
                self.add_module('dnn_' + str(i), nn.Linear(
                    in_features=self.input_size if i == 0 else self.hidden_size,
                    out_features=self.hidden_size
                ))
        gru_input_dim = self.input_size if self.dnn_layers == 0 else self.hidden_size

        for i in range(self.n_layers):
            self.add_module('pRNN_' + str(i), nn.GRU(
                input_size=gru_input_dim if i == 0 else self.hidden_size,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                bidirectional=self.bi,
                batch_first=True))

    def run_dnn(self, x):
        for i in range(self.dnn_layers):
            x = F.relu(getattr(self, 'dnn_'+str(i))(x))
        return x

    def run_pRNN(self, inputs, hidden, input_lengths):
        """

        :param input: (batch, seq_len, input_size)
        :param hidden: (num_layers * num_directions, batch, hidden_size)
        :return:
        """
        for i in range(self.n_layers):
            x = pack_padded_sequence(inputs, input_lengths, batch_first=True)
            output, hidden = getattr(self, 'pRNN_'+str(i))(x, hidden)
            output, _ = pad_packed_sequence(output, batch_first=True, padding_value=0.)
            hidden = hidden

            if self.bi:
                output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]

            if i < self.n_layers - 1:
                inputs, input_lengths = skip_add_pyramid(output, input_lengths, self.skip_add)

        return output, hidden, input_lengths

    def forward(self, inputs, hidden, input_lengths):
        if self.dnn_layers > 0:
            inputs = self.run_dnn(inputs)

        outputs, hidden, input_lengths = self.run_pRNN(inputs, hidden, input_lengths)

        if self.bi:
            hidden = torch.sum(hidden, 0)

        return outputs, hidden, input_lengths

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(2 if self.bi else 1, batch_size, self.hidden_size))
        if self.gpu:
            h0 = h0.cuda()
        return h0