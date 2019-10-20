from torchvision.models._utils import IntermediateLayerGetter
import torchvision.models.resnet as resnet

from torch.nn import functional as F

from collections import OrderedDict
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

squeeze = torchvision.models.squeezenet.squeezenet1_1(pretrained=True).eval()

class DeepSqueeze(torch.nn.Module):
    def __init__(self, num_classes):
        super(DeepSqueeze, self).__init__()
        backbone = squeeze.features

        classifier = DeepLabHead(512, num_classes)

        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = None

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x.half())

        result = OrderedDict()
#         x = features["out"]
        x = features
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return x
