from .base_vae import BaseVAE
from .leaps_vae import LeapsVAE
from .policy_vae import PolicyVAE
from .sketch_vae import SketchVAE

from dsl.production import Production
import torch

def load_model(model_cls_name: str, dsl: Production, device: torch.device) -> BaseVAE:
    model_cls = globals()[model_cls_name]
    assert issubclass(model_cls, BaseVAE)
    return model_cls(dsl, device)
