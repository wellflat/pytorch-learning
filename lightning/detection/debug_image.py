
from typing import Dict, Tuple
import random
from torch import Tensor
from torchvision.transforms import Compose, Normalize
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image

Batch = Tuple[Tuple[Tensor], Tuple[Dict[str, Tensor]], Tuple[int]]

def get_inv_transform() -> Compose:
    transforms = Compose([
        Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])
    return transforms

def display_image(
    writer: SummaryWriter,
    current_epoch: int,
    inputs: Tuple[Tensor],
    preds: Tuple[Dict[str, Tensor]],
    ids: Tuple[int]
) -> None:
    input_id = random.randrange(len(inputs))
    #np_image = inputs[input_id].detach().numpy().copy()
    inv_trans = get_inv_transform()
    inv_image = to_pil_image(inv_trans(inputs[input_id]))
    writer.add_image_with_boxes(
        f'{current_epoch}_{ids[input_id]}',
        inputs[input_id],
        preds[input_id]['boxes'],
        global_step=current_epoch
    )