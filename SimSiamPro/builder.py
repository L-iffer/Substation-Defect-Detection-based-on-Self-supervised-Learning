import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 1 if kernel_size == 3 else 3

        self.convsa = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.convsa(x)
        return self.sigmoid(x)

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        # self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        # self.encoder = nn.Sequential(*list((base_encoder(num_classes=dim, zero_init_residual=True)).children())[:-2])  # 丢弃了原始ResNet的最后的平均池化层和fc全连接层
        self.encoder = nn.Sequential()
        self.encoder.conv1 = list((base_encoder(num_classes=dim, zero_init_residual=True)).children())[0]
        self.encoder.bn1 = list((base_encoder(num_classes=dim, zero_init_residual=True)).children())[1]
        self.encoder.relu = list((base_encoder(num_classes=dim, zero_init_residual=True)).children())[2]
        self.encoder.maxpool = list((base_encoder(num_classes=dim, zero_init_residual=True)).children())[3]
        self.encoder.layer1 = list((base_encoder(num_classes=dim, zero_init_residual=True)).children())[4]
        self.encoder.layer2 = list((base_encoder(num_classes=dim, zero_init_residual=True)).children())[5]
        self.encoder.layer3 = list((base_encoder(num_classes=dim, zero_init_residual=True)).children())[6]
        self.encoder.layer4 = list((base_encoder(num_classes=dim, zero_init_residual=True)).children())[7]
        # build a 3-layer projector
        prev_dim = 2048

        # projector MLP
        self.encoder.fc1 = nn.Sequential(nn.Conv2d(in_channels=prev_dim, out_channels=prev_dim, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(prev_dim),
                                        nn.ReLU(inplace=True))
        self.encoder.sa1 = SpatialAttention()
        self.encoder.fc2 = nn.Sequential(nn.Conv2d(in_channels=prev_dim, out_channels=prev_dim, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(prev_dim),
                                        nn.ReLU(inplace=True))
        self.encoder.sa2 = SpatialAttention()
        self.encoder.fc3 = nn.Sequential(nn.Conv2d(in_channels=prev_dim, out_channels=dim, kernel_size=1, bias=True),
                                        nn.BatchNorm2d(dim, affine=False))
                                       
        self.encoder.fc3[0].bias.requires_grad = False  # stop-gradient

        # predictor MLP
        self.predictor = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=pred_dim, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(pred_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels=pred_dim, out_channels=dim, kernel_size=1, bias=True))
        

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        # z1 = self.encoder(x1) # NxC*H*W
        # z2 = self.encoder(x2) # NxC*H*W

        # p1 = self.predictor(z1) # NxC*H*W
        # p2 = self.predictor(z2) # NxC*H*W
        z1 = self.encoder.conv1(x1)
        z1 = self.encoder.bn1(z1)
        z1 = self.encoder.relu(z1)
        z1 = self.encoder.maxpool(z1)
        z1 = self.encoder.layer1(z1)
        z1 = self.encoder.layer2(z1)
        z1 = self.encoder.layer3(z1)
        z1 = self.encoder.layer4(z1)
        z1 = self.encoder.fc1(z1)
        z1 = self.encoder.sa1(z1) * z1  # SAM
        z1 = self.encoder.fc2(z1)
        z1 = self.encoder.sa2(z1) * z1  # SAM
        z1 = self.encoder.fc3(z1)

        z2 = self.encoder.conv1(x2)
        z2 = self.encoder.bn1(z2)
        z2 = self.encoder.relu(z2)
        z2 = self.encoder.maxpool(z2)
        z2 = self.encoder.layer1(z2)
        z2 = self.encoder.layer2(z2)
        z2 = self.encoder.layer3(z2)
        z2 = self.encoder.layer4(z2)
        z2 = self.encoder.fc1(z2)
        z2 = self.encoder.sa1(z2) * z2  # SAM
        z2 = self.encoder.fc2(z2)
        z2 = self.encoder.sa2(z2) * z2  # SAM
        z2 = self.encoder.fc3(z2)

        p1 = self.predictor(z1)

        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()
