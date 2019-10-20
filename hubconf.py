ependencies = ['torch']

from .semantic.DeepSqueeze.models import DeepSqueeze as DeepSqueeze_

# resnet18 is the name of entrypoint
def DeepSqueeze(pretrained=False, **kwargs):
    model = DeepSqueeze_(35)

    checkpoint = 'https://github.com/Sid1057/self-driving-cars-utils/blob/master/semantic/DeepSqueeze/DeepLab_v3_squeeze11_kitti_classes_iou_mean012_iou_max60.pth'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False)

    return model
