import logging

import timm
import torch
import torchvision
import torchvision.transforms as transforms

from robustbench.model_zoo.architectures.utils_architectures import normalize_model
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from typing import Union
from models import resnet26
from packaging import version

logger = logging.getLogger(__name__)


def get_torchvision_model(model_name: str, weight_version: str = "IMAGENET1K_V1"):
    """
    Restore a pre-trained model from torchvision
    Further details can be found here: https://pytorch.org/vision/0.14/models.html
    Input:
        model_name: Name of the model to create and initialize with pre-trained weights
        weight_version: Name of the pre-trained weights to restore
    Returns:
        model: The pre-trained model
        preprocess: The corresponding input pre-processing
    """
    assert version.parse(torchvision.__version__) >= version.parse("0.13"), "Torchvision version has to be >= 0.13"

    # check if the specified model name is available in torchvision
    available_models = torchvision.models.list_models(module=torchvision.models)
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not available in torchvision. Choose from: {available_models}")

    # get the weight object of the specified model and the available weight initialization names
    model_weights = torchvision.models.get_model_weights(model_name)
    available_weights = [init_name for init_name in dir(model_weights) if "IMAGENET1K" in init_name]

    # check if the specified type of weights is available
    if weight_version not in available_weights:
        raise ValueError(f"Weight type '{weight_version}' is not supported for torchvision model '{model_name}'."
                         f" Choose from: {available_weights}")

    # restore the specified weights
    model_weights = getattr(model_weights, weight_version)

    # setup the specified model and initialize it with the specified pre-trained weights
    model = torchvision.models.get_model(model_name, weights=model_weights)

    # get the transformation and add the input normalization to the model
    transform = model_weights.transforms()
    model = normalize_model(model, transform.mean, transform.std)
    logger.info(f"Successfully restored '{weight_version}' pre-trained weights"
                f" for model '{model_name}' from torchvision!")

    # create the corresponding input transformation
    preprocess = transforms.Compose([transforms.Resize(transform.resize_size, interpolation=transform.interpolation),
                                     transforms.CenterCrop(transform.crop_size),
                                     transforms.ToTensor()])
    return model, preprocess


def get_timm_model(model_name: str):
    """
    Restore a pre-trained model from timm: https://github.com/huggingface/pytorch-image-models/tree/main/timm
    Quickstart: https://huggingface.co/docs/timm/quickstart
    Input:
        model_name: Name of the model to create and initialize with pre-trained weights
    Returns:
        model: The pre-trained model
        preprocess: The corresponding input pre-processing
    """
    # check if the defined model name is supported as pre-trained model
    available_models = timm.list_models(pretrained=True)
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not available in timm. Choose from: {available_models}")

    # setup pre-trained model
    model = timm.create_model(model_name, pretrained=True)
    logger.info(f"Successfully restored the weights of '{model_name}' from timm.")

    # restore the input pre-processing
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config)

    # if there is an input normalization, add it to the model and remove it from the input pre-processing
    for transf in preprocess.transforms[::-1]:
        if isinstance(transf, transforms.Normalize):
            # add input normalization to the model
            model = normalize_model(model, mean=transf.mean, std=transf.std)
            preprocess.transforms.remove(transf)
            break

    return model, preprocess


def get_model(cfg, num_classes: int, device: Union[str, torch.device]):
    """
    Setup the pre-defined model architecture and restore the corresponding pre-trained weights
    Input:
        cfg: Configurations
        num_classes: Number of classes
        device: The device to put the loaded model
    Return:
        model: The pre-trained model
        preprocess: The corresponding input pre-processing
    """
    preprocess = None

    try:
        # load model from torchvision
        base_model, preprocess = get_torchvision_model(cfg.MODEL.ARCH, weight_version=cfg.MODEL.WEIGHTS)
    except ValueError:
        try:
            # load model from timm
            base_model, preprocess = get_timm_model(cfg.MODEL.ARCH)
        except ValueError:
            try:
                # load some custom models
                if cfg.MODEL.ARCH == "resnet26_gn":
                    base_model = resnet26.build_resnet26()
                    checkpoint = torch.load(cfg.MODEL.CKPT_PATH, map_location="cpu")
                    base_model.load_state_dict(checkpoint['net'])
                    base_model = normalize_model(base_model, resnet26.MEAN, resnet26.STD)
                else:
                    raise ValueError(f"Model {cfg.MODEL.ARCH} is not supported!")
                logger.info(f"Successfully restored model '{cfg.MODEL.ARCH}' from: {cfg.MODEL.CKPT_PATH}")
            except ValueError:
                # load model from robustbench
                if cfg.CORRUPTION.DATASET == 'ccc':
                    dataset_name = 'imagenet'
                else:
                    dataset_name = cfg.CORRUPTION.DATASET.split("_")[0]
                base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, dataset_name, ThreatModel.corruptions)

    return base_model.to(device), preprocess

