import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from config import Phase

'''
ResNet50 from torchvision
Val:
ImageClassification(
    crop_size=[224]
    resize_size=[232]
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    interpolation=InterpolationMode.BILINEAR
)'''
def get_transforms(phase: Phase) -> A.Compose:
    if phase == Phase.TRIAN:
        transforms = A.Compose([
            A.RandomResizedCrop(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  ## default mean and std
            ToTensorV2(p=1)
        ])
    elif phase == Phase.VAL:
        transforms = A.Compose([
            A.Resize(height=256, width=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            ToTensorV2(p=1)
        ])
    else:  ## test
        transforms = A.Compose([
            A.Resize(height=256, width=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            ToTensorV2(p=1)
        ])
    return transforms

## debug image for MLOps tools
def get_inv_transform() -> A.Compose:
    transforms = A.Compose([
        A.Normalize(mean=[ 0., 0., 0. ], std=[ 1/0.229, 1/0.224, 1/0.225 ]),
        A.Normalize(mean=[ -0.485, -0.456, -0.406 ], std=[ 1., 1., 1. ]),
    ])
    return transforms
