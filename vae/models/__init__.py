from .base_vae import BaseVAE
from .leaps_vae import LeapsVAE
from .policy_vae import PolicyVAE
from .sketch_vae import SketchVAE
from .double_vae import DoubleVAE
from .double_vae_translator import DoubleVAETranslator

from dsl import DSL
import torch

def load_model(model_cls_name: str, dsl: DSL, device: torch.device) -> BaseVAE:
    model_cls = globals()[model_cls_name]
    assert issubclass(model_cls, BaseVAE)
    return model_cls(dsl, device)
