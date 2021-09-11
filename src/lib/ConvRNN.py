from torch import zeros, cat, tanh, stack, sigmoid, split
from torch.nn import Module, Sequential, Conv2d, GroupNorm, Dropout2d

class CGRU_cell(Module):
    """
    ConvGRU Cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features, bias, no_bn, dropout, device):
        """
        Initialize the ConvLSTM cell
        :param shape: (int, int)
            Height and width of input tensor as (height, width).
        :param input_channels: int
            Number of channels of input tensor.
        :param filter_size: (int, int)
            Size of the convolutional kernel.
        :param num_features: int
            Number of channels of hidden state.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: str
            Whether or not to use cuda.
        """
        super(CGRU_cell, self).__init__()
        self.height, self.width = shape
        self.input_channels = input_channels
        # kernel_size of input_to_state equals state_to_state
        self.filter_size = filter_size
        self.num_features = num_features
        self.bias = bias
        self.device = device
        self.padding = (filter_size - 1) // 2
        self.no_bn = no_bn
        self.do_do = True if dropout > 0 else False
        self.dropout = Dropout2d(p=dropout)
        if self.no_bn:
            self.conv_gates = Sequential(
                Conv2d(in_channels=self.input_channels + self.num_features,
                          out_channels=2 * self.num_features,
                          kernel_size=self.filter_size,
                          stride=1,
                          padding=self.padding,
                          bias=self.bias),
                )
            self.conv_can = Sequential(
                Conv2d(in_channels=self.input_channels + self.num_features,
                          out_channels=self.num_features,
                          kernel_size=self.filter_size,
                          stride=1,
                          padding=self.padding,
                          bias=self.bias),
                )
        else:
            self.conv_gates = Sequential(
                Conv2d(in_channels=self.input_channels + self.num_features,
                          out_channels=2 * self.num_features,
                          kernel_size=self.filter_size,
                          stride=1,
                          padding=self.padding,
                          bias=self.bias),
                GroupNorm(num_groups=2 * self.num_features // 1,
                             num_channels=2 * self.num_features)
                )
            self.conv_can = Sequential(
                Conv2d(in_channels=self.input_channels + self.num_features,
                          out_channels=self.num_features,
                          kernel_size=self.filter_size,
                          stride=1,
                          padding=self.padding,
                          bias=self.bias),
                GroupNorm(num_groups=self.num_features // 1,
                             num_channels=self.num_features)
                )

    def move_device(self, device):
        self.device = device

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        if hidden_state is None:
            htprev = zeros(inputs.size(1), self.num_features,
                                 self.height, self.width).to(self.device, non_blocking=True)
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = zeros(htprev.size(0), self.input_channels,
                                self.height, self.width).to(self.device, non_blocking=True)
            else:
                x = inputs[index, ...]

            # W * (X_t + H_t-1)
            gates = self.conv_gates(cat([x, htprev], dim=1))

            # activate gates
            gamma, beta = split(gates, self.num_features, dim=1)
            reset_gate = sigmoid(gamma)
            update_gate = sigmoid(beta)

            # h' = tanh(W*(x+reset_gate*H_t-1))
            ht = tanh(self.conv_can(cat([x, reset_gate * htprev], dim=1)))
            htnext = (1 - update_gate) * htprev + update_gate * ht
            # drop out
            if self.do_do:
              htnext = self.dropout(htnext)
            output_inner.append(htnext)
            htprev = htnext
        return stack(output_inner), htnext


class CLSTM_cell(Module):
    """ConvLSTMCell
    """
    def __init__(self, shape, input_channels, filter_size, num_features, bias, no_bn, device):
        """
        Initialize the ConvLSTM cell
        :param shape: (int, int)
            Height and width of input tensor as (height, width).
        :param input_channels: int
            Number of channels of input tensor.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_features: int
            Number of channels of hidden state.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: str
            Whether or not to use cuda.
        """
        super(CLSTM_cell, self).__init__()
        self.height, self.width = shape
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.bias = bias
        self.device = device
        self.padding = (filter_size - 1) // 2
        self.no_bn = no_bn
        if self.no_bn:
            self.conv = Sequential(
                Conv2d(in_channels=self.input_channels + self.num_features,
                          out_channels=4 * self.num_features,
                          kernel_size=self.filter_size,
                          stride=1,
                          padding=self.padding,
                          bias=self.bias),
                )
        else:
            self.conv = Sequential(
                Conv2d(in_channels=self.input_channels + self.num_features,
                          out_channels=4 * self.num_features,
                          kernel_size=self.filter_size,
                          stride=1,
                          padding=self.padding,
                          bias=self.bias),
                GroupNorm(num_groups=4 * self.num_features // 2,
                             num_channels=4 * self.num_features)
                )

    def move_device(self, device):
        self.device = device

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        if hidden_state is None:
            hx = zeros(inputs.size(1), self.num_features, self.height, self.width).to(self.device, non_blocking=True)
            cx = zeros(inputs.size(1), self.num_features, self.height, self.width).to(self.device, non_blocking=True)
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = zeros(hx.size(0), self.input_channels, self.height, self.width).to(self.device, non_blocking=True)
            else:
                x = inputs[index, ...]

            ingate, forgetgate, cellgate, outgate = split(self.conv(cat([x, hx], dim=1)), self.num_features, dim=1)
            ingate = sigmoid(ingate)
            forgetgate = sigmoid(forgetgate)
            cellgate = tanh(cellgate)
            outgate = sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return stack(output_inner), (hy, cy)
