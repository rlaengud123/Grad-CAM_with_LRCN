import torch.nn as nn
from torchvision import models


class ConvLstm(nn.Module):
    def __init__(self, latent_dim, model):
        super(ConvLstm, self).__init__()
        self.conv_model = Pretrained_conv(latent_dim, model)

    def forward(self, x):
        batch_size = 1
        timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.conv_model(conv_input)
        return conv_output


class Pretrained_conv(nn.Module):
    def __init__(self, latent_dim, model):
        if model == 'resnet152':
            super(Pretrained_conv, self).__init__()
            self.conv_model = models.resnet152(pretrained=True)
            # ====== freezing all of the layers ======
            for param in self.conv_model.parameters():
                param.requires_grad = False
            # ====== changing the last FC layer to an output with the size we need. this layer is un freezed ======
            self.conv_model.fc = nn.Linear(
                self.conv_model.fc.in_features, latent_dim)
        elif model == 'densenet201':
            super(Pretrained_conv, self).__init__()
            self.conv_model = models.densenet201(pretrained=True)
            # ====== freezing all of the layers ======
            for param in self.conv_model.parameters():
                param.requires_grad = False
            # ====== changing the last FC layer to an output with the size we need. this layer is un freezed ======
            self.conv_model.classifier = nn.Linear(
                self.conv_model.classifier.in_features, latent_dim)
        elif model == 'densenet161':
            super(Pretrained_conv, self).__init__()
            self.conv_model = models.densenet161(pretrained=True)
            # ====== freezing all of the layers ======
            for param in self.conv_model.parameters():
                param.requires_grad = False
            # ====== changing the last FC layer to an output with the size we need. this layer is un freezed ======
            self.conv_model.classifier = nn.Linear(
                self.conv_model.classifier.in_features, latent_dim)

    def forward(self, x):
        return self.conv_model(x)
