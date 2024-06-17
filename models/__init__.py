from .ias_model import IASModel
from .lbp_model import LBPModel
from .resnet_model import ResNetModel


MODEL_DICT = {
    'ias_model': IASModel,
    'lbp_model': LBPModel,
    'resnet_model': ResNetModel
}