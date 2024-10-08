import torch
from torch import nn
from torch.hub import load_state_dict_from_url

class VGG19(nn.Module):
    def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=1000, progress=True):
        super(VGG19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_classes = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights is not None:
            if init_weights == 'vgg19':
                # progress (bool): If True, displays a progress bar of the download to stderr
                model_url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
                state_dict = load_state_dict_from_url(model_url, progress=progress)
            else:
                state_dict = torch.load(init_weights)
            self.load_state_dict(state_dict)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x, layers_to_save=None):
        outputs = {}  # Dictionary to store outputs of specified layers
        if self.feature_mode:
            module_list = list(self.features.modules())
            for idx, l in enumerate(module_list[1:27]):  # conv4_4
                x = l(x)
                if layers_to_save and idx in layers_to_save:
                    outputs[f'layer_{idx}'] = x  # Save the output of the specified layer
        if not self.feature_mode:
            for idx, l in enumerate(self.features):
                x = l(x)
                if layers_to_save and idx in layers_to_save:
                    outputs[f'layer_{idx}'] = x  # Save the output of the specified layer
            x = x.view(x.size(0), -1)
            for idx, l in enumerate(self.classifier):
                x = l(x)
                classifier_idx = idx + len(self.features)  # Continue layer indexing
                if layers_to_save and classifier_idx in layers_to_save:
                    outputs[f'layer_{classifier_idx}'] = x  # Save the output of the classifier layer

        if layers_to_save:
            return outputs, x  # Return both the intermediate outputs and final result
        return x  # Only return the final output if no layers are specified
