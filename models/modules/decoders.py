import torch
import torch.nn as nn
import torch.nn.functional as F

from models.helpers import mask_3d


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.batch_size = config["batch_size"]
        self.hidden_size = config["decoder_hidden"]
        embedding_dim = config.get("embedding_dim", None)
        self.embedding_dim = embedding_dim if embedding_dim is not None else self.hidden_size
        self.embedding = nn.Embedding(config.get("n_classes", 32), self.embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(
            input_size=self.embedding_dim+self.hidden_size if config['decoder'].lower() == 'bahdanau' else self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=config.get("decoder_layers", 1),
            dropout=config.get("decoder_dropout", 0),
            bidirectional=config.get("bidirectional_decoder", False),
            batch_first=True)
        if config['decoder'] != "RNN":
            self.attention = Attention(
                self.batch_size,
                self.hidden_size,
                method=config.get("attention_score", "dot"),
                mlp=config.get("attention_mlp_pre", False))

        self.gpu = config.get("gpu", False)
        self.decoder_output_fn = F.log_softmax if config.get('loss', 'NLL') == 'NLL' else None

    def forward(self, **kwargs):
        """ Must be overrided """
        raise NotImplementedError


class RNNDecoder(Decoder):
    def __init__(self, config):
        super(RNNDecoder, self).__init__(config)
        self.output_size = config.get("n_classes", 32)
        self.character_distribution = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, **kwargs):
        input = kwargs["input"]
        hidden = kwargs["hidden"]
        # RNN (Eq 7 paper)
        embedded = self.embedding(input).unsqueeze(0)
        rnn_input = torch.cat((embedded, hidden.unsqueeze(0)), 2)  # NOTE : Tf concats `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
        # rnn_output, rnn_hidden = self.rnn(rnn_input.transpose(1, 0), hidden.unsqueeze(0))
        rnn_output, rnn_hidden = self.rnn(embedded.transpose(1, 0), hidden.unsqueeze(0))
        output = rnn_output.squeeze(1)
        output = self.character_distribution(output)

        if self.decoder_output_fn:
            output = self.decoder_output_fn(output, -1)

        return output, rnn_hidden.squeeze(0)


class BahdanauDecoder(Decoder):
    """
        Corresponds to BahdanauAttnDecoderRNN in Pytorch tuto
    """

    def __init__(self, config):
        super(BahdanauDecoder, self).__init__(config)
        self.output_size = config.get("n_classes", 32)
        self.character_distribution = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, **kwargs):
        """

        :param input: [B]
        :param prev_context: [B, H]
        :param prev_hidden: [B, H]
        :param encoder_outputs: [B, T, H]
        :return: output (B), context (B, H), prev_hidden (B, H), weights (B, T)
        """

        input = kwargs["input"]
        prev_hidden = kwargs["prev_hidden"]
        encoder_outputs = kwargs["encoder_outputs"]
        seq_len = kwargs.get("seq_len", None)

        # check inputs
        assert input.size() == torch.Size([self.batch_size])
        assert prev_hidden.size() == torch.Size([self.batch_size, self.hidden_size])

        # Attention weights
        weights = self.attention.forward(prev_hidden, encoder_outputs, seq_len)  # B x T
        context = weights.unsqueeze(1).bmm(encoder_outputs).squeeze(1)  # [B x H]

        # embed characters
        embedded = self.embedding(input).unsqueeze(0)
        assert embedded.size() == torch.Size([1, self.batch_size, self.embedding_dim])

        rnn_input = torch.cat((embedded, context.unsqueeze(0)), 2)

        outputs, hidden = self.rnn(rnn_input.transpose(1, 0), prev_hidden.unsqueeze(0)) # 1 x B x N, B x N

        # output = self.proj(torch.cat((outputs.squeeze(0), context), 1))
        output = self.character_distribution(outputs.squeeze(0))

        if self.decoder_output_fn:
            output = self.decoder_output_fn(output, -1)

        if len(output.size()) == 3:
            output = output.squeeze(1)

        return output, hidden.squeeze(0), weights


class LuongDecoder(Decoder):
    """
        Corresponds to AttnDecoderRNN
    """

    def __init__(self, config):
        super(LuongDecoder, self).__init__(config)
        self.output_size = config.get("n_classes", 32)
        self.character_distribution = nn.Linear(2*self.hidden_size, self.output_size)

    def forward(self, **kwargs):
        """

        :param input: [B]
        :param prev_context: [B, H]
        :param prev_hidden: [B, H]
        :param encoder_outputs: [B, T, H]
        :return: output (B, V), context (B, H), prev_hidden (B, H), weights (B, T)

        https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
        TF says : Perform a step of attention-wrapped RNN.
        - Step 1: Mix the `inputs` and previous step's `attention` output via
          `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
          `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
          and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).
        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time step.
          state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.
        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:
          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `AttentionWrapperState`
             containing the state calculated at this time step.
        Raises:
          TypeError: If `state` is not an instance of `AttentionWrapperState`.

        """
        input = kwargs["input"]
        prev_hidden = kwargs["prev_hidden"]
        encoder_outputs = kwargs["encoder_outputs"]
        seq_len = kwargs.get("seq_len", None)

        # RNN (Eq 7 paper)
        embedded = self.embedding(input).unsqueeze(1) # [B, H]
        prev_hidden = prev_hidden.unsqueeze(0)
        # rnn_input = torch.cat((embedded, prev_context), -1) # NOTE : Tf concats `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
        # rnn_output, hidden = self.rnn(rnn_input.transpose(1, 0), prev_hidden)
        rnn_output, hidden = self.rnn(embedded, prev_hidden)
        rnn_output = rnn_output.squeeze(1)

        # Attention weights (Eq 6 paper)
        weights = self.attention.forward(rnn_output, encoder_outputs, seq_len) # B x T
        context = weights.unsqueeze(1).bmm(encoder_outputs).squeeze(1)  # [B x N]

        # Projection (Eq 8 paper)
        # /!\ Don't apply tanh on outputs, it fucks everything up
        output = self.character_distribution(torch.cat((rnn_output, context), 1))

        # Apply log softmax if loss is NLL
        if self.decoder_output_fn:
            output = self.decoder_output_fn(output, -1)

        if len(output.size()) == 3:
            output = output.squeeze(1)

        return output, hidden.squeeze(0), weights


class Attention(nn.Module):
    """
    Inputs:
        last_hidden: (batch_size, hidden_size)
        encoder_outputs: (batch_size, max_time, hidden_size)
    Returns:
        attention_weights: (batch_size, max_time)
    """
    def __init__(self, batch_size, hidden_size, method="dot", mlp=False):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if method == 'dot':
            pass
        elif method == 'general':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
        elif method == 'bahdanau':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
        else:
            raise NotImplementedError

        self.mlp = mlp
        if mlp:
            self.phi = nn.Linear(hidden_size, hidden_size, bias=False)
            self.psi = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, last_hidden, encoder_outputs, seq_len=None):
        batch_size, seq_lens, _ = encoder_outputs.size()
        if self.mlp:
            last_hidden = self.phi(last_hidden)
            encoder_outputs = self.psi(encoder_outputs)

        attention_energies = self._score(last_hidden, encoder_outputs, self.method)
        # attn_energies = Variable(torch.zeros(batch_size, seq_lens))  # B x S

        if seq_len is not None:
            attention_energies = mask_3d(attention_energies, seq_len, -float('inf'))

        return F.softmax(attention_energies, -1)

    def _score(self, last_hidden, encoder_outputs, method):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)
        :param method: str (`dot`, `general`, `concat`)
        :return:
        """

        # assert last_hidden.size() == torch.Size([batch_size, self.hidden_size]), last_hidden.size()
        assert encoder_outputs.size()[-1] == self.hidden_size

        if method == 'dot':
            last_hidden = last_hidden.unsqueeze(-1)
            return encoder_outputs.bmm(last_hidden).squeeze(-1)

        elif method == 'general':
            x = self.Wa(last_hidden)
            x = x.unsqueeze(-1)
            return encoder_outputs.bmm(x).squeeze(-1)

        elif method == "concat":
            x = last_hidden.unsqueeze(1)
            x = F.tanh(self.Wa(torch.cat((x, encoder_outputs), 1)))
            return x.bmm(self.va.unsqueeze(2)).squeeze(-1)

        elif method == "bahdanau":
            x = last_hidden.unsqueeze(1)
            out = F.tanh(self.Wa(x) + self.Ua(encoder_outputs))
            return out.bmm(self.va.unsqueeze(2)).squeeze(-1)

        else:
            raise NotImplementedError

    def extra_repr(self):
        return 'score={}, mlp_preprocessing={}'.format(
            self.method, self.mlp)
