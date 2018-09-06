import models
from torchsummary import summary

linknet = models.UNet11(num_classes=11, num_filters=32, pretrained=True)
linknet.cuda()
summary(linknet, input_size=(3, 1024, 1280))
