ependencies = ['torch']

from semantic.DeepSqueeze.models import DeepSqueeze as DeepSqueeze_

# resnet18 is the name of entrypoint
def DeepSqueeze(pretrained=False, **kwargs):
    model = DeepSqueeze(35)
    model.load_state_dict(torch.load('semantic/DeepSqueeze/models/DeepLab_v3_squeeze11_kitti_classes_iou_mean012_iou_max60.pth'))

    return model
