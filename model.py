import torch
import torch.nn as nn

class ShallowNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ShallowNet, self).__init__()
        self.MLP = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.Tanh(),
                                   nn.Linear(hidden_size, output_size))
        
        self.apply(self._init_weights) # weight initialization

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # nn.init.xavier_uniform_(module.weight.data)
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.MLP(x)

class DeepNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepNet, self).__init__()
        self.MLP = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.Tanh(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.Tanh(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.Tanh(),
                                   nn.Linear(hidden_size, output_size)
                                   )
        
        self.apply(self._init_weights) # weight initialization

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            # module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.MLP(x)
