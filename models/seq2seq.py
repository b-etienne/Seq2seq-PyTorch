import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from utils.misc import check_size
from torch.autograd import Variable
from models.modules.encoders import EncoderRNN, EncoderPyRNN
from models.modules.decoders import BahdanauDecoder, LuongDecoder, RNNDecoder
from models.helpers import mask_3d


class Seq2Seq(nn.Module):
    """
        Sequence to sequence module
    """

    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        self.SOS = config.get("start_index", 1),
        self.vocab_size = config.get("n_classes", 32)
        self.batch_size = config.get("batch_size", 1)
        self.sampling_prob = config.get("sampling_prob", 0.)
        self.gpu = config.get("gpu", False)

        # Encoder
        if config["encoder"] == "PyRNN":
            self._encoder_style = "PyRNN"
            self.encoder = EncoderPyRNN(config)
        else:
            self._encoder_style = "RNN"
            self.encoder = EncoderRNN(config)

        # Decoder
        self.use_attention = config["decoder"] != "RNN"
        if config["decoder"] == "Luong":
            self.decoder = LuongDecoder(config)
        elif config["decoder"] == "Bahdanau":
            self.decoder = BahdanauDecoder(config)
        else:
            self.decoder = RNNDecoder(config)

        if config.get('loss') == 'cross_entropy':
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
            config['loss'] = 'cross_entropy'
        else:
            self.loss_fn = torch.nn.NLLLoss(ignore_index=0)
            config['loss'] = 'NLL'
        self.loss_type = config['loss']
        print(config)

    def encode(self, x, x_len):

        batch_size = x.size()[0]
        init_state = self.encoder.init_hidden(batch_size)
        if self._encoder_style == "PyRNN":
            encoder_outputs, encoder_state, input_lengths = self.encoder.forward(x, init_state, x_len)
        else:
            encoder_outputs, encoder_state = self.encoder.forward(x, init_state, x_len)

        assert encoder_outputs.size()[0] == self.batch_size, encoder_outputs.size()
        assert encoder_outputs.size()[-1] == self.decoder.hidden_size

        if self._encoder_style == "PyRNN":
            return encoder_outputs, encoder_state.squeeze(0), input_lengths
        return encoder_outputs, encoder_state.squeeze(0)

    def decode(self, encoder_outputs, encoder_hidden, targets, targets_lengths, input_lengths):
        """
        Args:
            encoder_outputs: (B, T, H)
            encoder_hidden: (B, H)
            targets: (B, L)
            targets_lengths: (B)
            input_lengths: (B)
        Vars:
            decoder_input: (B)
            decoder_context: (B, H)
            hidden_state: (B, H)
            attention_weights: (B, T)
        Outputs:
            alignments: (L, T, B)
            logits: (B*L, V)
            labels: (B*L)
        """

        batch_size = encoder_outputs.size()[0]
        max_length = targets.size()[1]
        # decoder_attns = torch.zeros(batch_size, MAX_LENGTH, MAX_LENGTH)
        decoder_input = Variable(torch.LongTensor([self.SOS] * batch_size)).squeeze(-1)
        decoder_context = encoder_outputs.transpose(1, 0)[-1]
        decoder_hidden = encoder_hidden

        alignments = Variable(torch.zeros(max_length, encoder_outputs.size(1), batch_size))
        logits = Variable(torch.zeros(max_length, batch_size, self.decoder.output_size))

        if self.gpu:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()
            logits = logits.cuda()

        for t in range(max_length):

            # The decoder accepts, at each time step t :
            # - an input, [B]
            # - a context, [B, H]
            # - an hidden state, [B, H]
            # - encoder outputs, [B, T, H]

            check_size(decoder_input, self.batch_size)
            check_size(decoder_hidden, self.batch_size, self.decoder.hidden_size)

            # The decoder outputs, at each time step t :
            # - an output, [B]
            # - a context, [B, H]
            # - an hidden state, [B, H]
            # - weights, [B, T]

            if self.use_attention:
                check_size(decoder_context, self.batch_size, self.decoder.hidden_size)
                outputs, decoder_hidden, attention_weights = self.decoder.forward(
                    input=decoder_input.long(),
                    prev_hidden=decoder_hidden,
                    encoder_outputs=encoder_outputs,
                    seq_len=input_lengths)
                alignments[t] = attention_weights.transpose(1, 0)
            else:
                outputs, hidden = self.decoder.forward(
                    input=decoder_input.long(),
                    hidden=decoder_hidden)

            # print(outputs[0])
            logits[t] = outputs

            use_teacher_forcing = random.random() > self.sampling_prob

            if use_teacher_forcing and self.training:
                decoder_input = targets[:, t]

            # SCHEDULED SAMPLING
            # We use the target sequence at each time step which we feed in the decoder
            else:
                # TODO Instead of taking the direct one-hot prediction from the previous time step as the original paper
                # does, we thought it is better to feed the distribution vector as it encodes more information about
                # prediction from previous step and could reduce bias.
                topv, topi = outputs.data.topk(1)
                decoder_input = topi.squeeze(-1).detach()


        labels = targets.contiguous().view(-1)

        if self.loss_type == 'NLL': # ie softmax already on outputs
            mask_value = -float('inf')
            print(torch.sum(logits, dim=2))
        else:
            mask_value = 0

        logits = mask_3d(logits.transpose(1, 0), targets_lengths, mask_value)
        logits = logits.contiguous().view(-1, self.vocab_size)

        return logits, labels.long(), alignments

    @staticmethod
    def custom_loss(logits, labels):

        # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = 0
        mask = (labels > tag_pad_token).float()

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).data[0])

        # pick the values for the label and zero out the rest with the mask
        logits = logits[range(logits.shape[0]), labels] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = -torch.sum(logits) / nb_tokens

        return ce_loss

    def step(self, batch):
        x, y, x_len, y_len = batch
        if self.gpu:
            x = x.cuda()
            y = y.cuda()
            x_len = x_len.cuda()
            y_len = y_len.cuda()

        if self._encoder_style == "PyRNN":
            encoder_out, encoder_state, x_len = self.encode(x, x_len)
        else:
            encoder_out, encoder_state = self.encode(x, x_len)
        logits, labels, alignments = self.decode(encoder_out, encoder_state, y, y_len, x_len)
        return logits, labels, alignments

    def loss(self, batch):
        logits, labels, alignments = self.step(batch)
        loss = self.loss_fn(logits, labels)
        # loss2 = self.custom_loss(logits, labels)
        return loss, logits, labels, alignments

