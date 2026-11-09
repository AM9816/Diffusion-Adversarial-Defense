
from diffusers import DiffusionPipeline
import torchattacks
import torch
from torch import nn, Tensor as T, functional as F
import torchvision
from torchvision import transforms
import os

from huggingface_hub import hf_hub_download

from transformers import AutoImageProcessor, AutoModelForImageClassification




from transformers import AutoModelForImageClassification, pipeline as tpipe

from torch_common_clean import *

vit_preproc = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])

cnn_preproc = torchvision.transforms.Normalize(
    [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])

vit_preproc_mnist = transforms.Compose([

    transforms.Resize((224, 224)),
    # transforms.Grayscale(num_output_channels=3),
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, .5, .5], std=[0.5, .5, .5])
])


vit_preproc_fashion_mnist = transforms.Compose([

    transforms.Resize((224, 224)),
    # transforms.Grayscale(num_output_channels=3),
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


sched_conf_path = r"ddpm_scheduler_config_revised.json"



class PytorchWrapper(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f=f;
    def forward(self ,x):
        return self.f(x)


def list_CIFAR10_classifier_names():
    print(torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True))

def get_CIFAR10_classifier(name="cifar10_resnet56", device=gpu):
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", 
        name, pretrained=True)
    return model.to(device).eval()


class _transformer(nn.Module):
    def __init__(self, model, _expand_3c=True): 
        super().__init__()
        self.m=model
        self._expand = _expand_3c
    def forward(self, x, *args, **kw): 
        x = x.expand(-1, 3, -1, -1) if self._expand else x
        return self.m(x, *args, **kw).logits



def get_MNIST_classifier(device=gpu):
    model_id = "farleyknight-org-username/vit-base-mnist"
    model = tpipe("image-classification", model=model_id).model
    model = get_normalized_classifier(model, vit_preproc_mnist)
    return _transformer(model).to(device).eval()

def get_CIAFAR10_diffusion_model(device=gpu):
    ddpm = (pipe := DiffusionPipeline.from_pretrained(
        "google/ddpm-cifar10-32", 
        device_map="cuda" if device==gpu else "cpu")).unet
    return ddpm.to(device).eval();





def get_MNIST_diffusion_model(device=gpu):
    ddpm = (pipe := DiffusionPipeline.from_pretrained(
            "1aurent/ddpm-mnist",
            device_map="cuda" 
        )).unet
    
    # return _to_colour(ddpm.eval()).to(device)
    return (ddpm.eval()).to(device)

def get_FMNIST_classifier(device=gpu):
    # model = torch.hub.load('harshhmaniya/Fashion_MNIST_Classification_Models', 'model', source='hf')
    # processor = AutoImageProcessor.from_pretrained("thilinadj/image_classification_tdj_fashion-mnist")
    model = AutoModelForImageClassification.from_pretrained("thilinadj/image_classification_tdj_fashion-mnist")
    return _transformer(model.to(device).eval())

def get_FMNIST_diffusion_model(device=gpu):
    return get_MNIST_diffusion_model(device)
    # return _transformer(get_MNIST_diffusion_model(device), False)
    # ddpm_unet = (pipe := DiffusionPipeline.from_pretrained(
    #         "nielsrogge/ddpm-fashion-mnist", 
    #         device_map="cuda"
    #     )).unet
    # return ddpm_unet.eval().to(device)
