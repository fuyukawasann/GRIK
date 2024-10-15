import torch
import torch.nn as nn
import torch.optim as optim

class CTCBeamSearchDecoder (nn.Module):
    def __init__(self, num_classes):
        super(CTCBeamSearchDecoder, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits):
        pass

        