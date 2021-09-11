from torch import nn, mean

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def move_device(self, device):
        self.encoder.to(device)
        self.encoder.move_device(device)
        
        self.decoder.to(device)
        self.decoder.move_device(device)

    def encode(self, inputs):
        code = self.encoder.get_code(inputs)
        if isinstance(code, tuple): # LSTM
            code = code[0]
        return mean(code,dim=0).unsqueeze(0)

    def forward(self, inputs):
        state = self.encoder(inputs)
        output = self.decoder(state)
        return output
