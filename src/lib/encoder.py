from torch import nn, reshape
from lib.utils import make_layers


class Encoder(nn.Module):
    def __init__(self, subnets, rnns, seq_len=10):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)
        self.seq_len = seq_len

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def move_device(self, device):
        for i in range(1, self.blocks + 1):
            getattr(self, 'rnn' + str(i)).move_device(device)

    def forward_by_stage(self, inputs, state, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        outputs_stage, state_stage = rnn(inputs, hidden_state=state, seq_len=self.seq_len)
        return outputs_stage, state_stage

    def forward(self, inputs):
        # to S,B,1,64,64
        next_inputs = inputs.transpose(0, 1)
    
        hidden_states = []
        for i in range(1, self.blocks + 1):
            next_inputs, state_stage = self.forward_by_stage(inputs=next_inputs,
                                                             state=None,
                                                             subnet=getattr(self, 'stage' + str(i)),
                                                             rnn=getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)

    def get_code(self, inputs):
        next_inputs = inputs.transpose(0, 1)
    
        for i in range(1, self.blocks + 1):
            next_inputs, state_stage = self.forward_by_stage(inputs=next_inputs,
                                                             state=None,
                                                             subnet=getattr(self, 'stage' + str(i)),
                                                             rnn=getattr(self, 'rnn' + str(i)))
        return state_stage
