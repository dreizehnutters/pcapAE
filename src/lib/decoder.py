from torch import nn, reshape
from lib.utils import make_layers


class Decoder(nn.Module):
    def __init__(self, subnets, rnns, seq_len=10):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)
        self.seq_len = seq_len

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))
    
    def move_device(self, device):
        for i in range(1, self.blocks + 1):
            getattr(self, 'rnn' + str(i)).move_device(device)

    def forward_by_stage(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, hidden_state=state, seq_len=self.seq_len)
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        return inputs

    def forward(self, hidden_states):
        inputs = self.forward_by_stage(inputs=None,
                                       state=hidden_states[-1],
                                       subnet=getattr(self, f'stage{self.blocks}'),
                                       rnn=getattr(self, f'rnn{self.blocks}'))
        for i in list(range(1, self.blocks))[::-1]:
            inputs = self.forward_by_stage(inputs=inputs, 
                                           state=hidden_states[i - 1],
                                           subnet=getattr(self, 'stage' + str(i)),
                                           rnn=getattr(self, 'rnn' + str(i)))
        # to B,S,1,D,D
        inputs = inputs.transpose(0, 1)
        return inputs
