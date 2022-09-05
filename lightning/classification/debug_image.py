import math
import random
import numpy as np
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Normalize
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt


def get_inv_transform() -> Compose:
    transforms = Compose([
        Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])
    return transforms

def display_image(writer: SummaryWriter, current_epoch: int, inputs: Tensor, probas: Tensor, preds: Tensor, batch_idx: int) -> None:
    #writer: SummaryWriter = self.logger.experiment
    input_id = random.randrange(len(inputs))
    #np_image = inputs[input_id].detach().numpy().copy()
    inv_trans = get_inv_transform()
    inv_image = to_pil_image(inv_trans(inputs[input_id]))
    pred_class = preds[input_id].item()
    pred_proba = math.floor(probas[input_id][preds[input_id]].item()*1e+3) / 1e+3
    plt.imshow(inv_image)
    plt.title(f'{current_epoch}: {pred_class}')
    plt.xlabel(f"score: {pred_proba}")
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.show()