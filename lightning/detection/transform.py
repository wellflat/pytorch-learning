import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from config import Phase

def get_transforms(phase: Phase) -> A.Compose:
    if phase == Phase.TRIAN:
        transforms = A.Compose([
            A.Resize(height=512, width=512, p=1),
            #A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  ## default mean and std
            #A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
            ToTensorV2(p=1)
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ))
    elif phase == Phase.VAL:
        transforms = A.Compose([
            A.Resize(height=512, width=512, p=1),
            ToTensorV2(p=1)
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ))
    else:  ## test
        transforms = A.Compose([
            A.Resize(height=512, width=512, p=1),
            ToTensorV2(p=1)
        ])
    return transforms