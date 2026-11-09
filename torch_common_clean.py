
import torch 
from torch import nn, Tensor as T
import torchvision
from torchvision.datasets import CIFAR10, ImageFolder, MNIST, FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.utils import make_grid

import numpy as np
import random
from time import time as now



cpu, gpu = torch.device("cpu"), torch.device("cuda")

en = enumerate
islist = lambda l: isinstance(l, list) or isinstance(l, tuple)

LOW_VRAM = True
EPSILON = 1e-8
BPDA_TYPE = None #"full"
MAX_t = 100


def randint(low, high):
    return torch.randint(low, high, (1,))[0]

def project_linf(x, orig, eps):
    diff = -(x-orig)
    return orig + torch.sign(diff) * torch.min(diff.abs(), torch.ones_like(diff) * eps)

def mask_nans(x, orig):
    # x = x[torch.isnan(x)] = orig
    return x



def project_l2(x : T, orig : T, eps) -> T:
    pass;

    delta = x - orig
    delta_norms = torch.norm(delta.view(x.shape[0], -1), p=2, dim=1)
    factor = eps / (EPSILON+delta_norms)
    factor = torch.min(factor, torch.ones_like(delta_norms))
    delta = delta * factor.view(-1, 1, 1, 1)

    return torch.clamp(orig + delta, min=0, max=1)#.detach()







def get_acc(out : T, y : T):
    out = torch.argmax(out, dim=1)
    return (out.cpu() == y.cpu()).float().mean()


def topil(tensor : T, rescale = True):
    #if rescale: tensor = ((tensor + 1.) / 2.)
    tensor = tensor.cpu().detach()
    #tensor = tempScale(tensor)
    return transforms.ToPILImage()(
        tensor if len(list(tensor.shape)) == 3 else
        make_grid([img for img in tensor], pad_value=0)
    )

def showtensor(x):
    topil(x.cpu()).show()

def showtensors(*xs):
    if len(xs) > 0 and islist(xs[0]): xs=xs[0]
    out = torch.cat([x.cpu() for x in xs],dim=0)
    showtensor(out)



def get_train_test_split(dataset, split, bsize = 32, doShuffle = True, _N=None):
    
    if _N is not None:
        dataset = Subset(dataset, range(_N))
    
    if split is None:
        return DataLoader(dataset, bsize, doShuffle)

    size = len(dataset)
    testSize = split * size
    trainSize = size - testSize

    return [
        # the dataLOADER created from the dataSET returned by .. 
        DataLoader(dataSet, bsize, doShuffle) 
            for dataSet in 

                # .. randomly splitting the whole dataset
                torch.utils.data.random_split(
                                    dataset, 
                                    [int(trainSize), int(testSize)])
    ]

import kagglehub


def getFashionMNIST(
        bsize = 64,
        shuffle=False
    ):
    
    transform = torchvision.transforms.ToTensor()

    train_data = FashionMNIST(
        root="data",       
        train=True,         
        download=True,      
        transform=transform  
    )

    return DataLoader(train_data, bsize, shuffle=shuffle)


def getTinyImgnet(
    targetImgSize=64, bsize = 32,
    shuffle=True,
    train_test_split=None, _N=None):

    tforms = torchvision.transforms.Compose([
        transforms.Resize(targetImgSize),
        transforms.ToTensor()
    ])

    imgnet = kagglehub.dataset_download("akash2sharma/tiny-imagenet")
    imgnet = ImageFolder(imgnet)
    print(imgnet)

# getTinyImgnet(); exit()


def getMNIST(
    dir="mnist",
    imgsize = 28,
    bsize = 32):

    tforms = torchvision.transforms.Compose([
        transforms.Resize(imgsize),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])


    mnist = MNIST(dir, transform=tforms,
                  download=True)

    return DataLoader(mnist, bsize, False)



def getCIFAR(
        dir="cifar",
        targetImgSize = 32, bsize = 32,
        # download if dataset is not already downloaded 
        # in folder specified by dir
        downloadIfNotExists = True,
        shuffle = True,
        train_test_split = None,
        _N=None,
    ):

    tforms = transforms.Compose([    
        transforms.Resize(targetImgSize), transforms.ToTensor(),
        #transforms.Normalize(
        #    mean=[0.4914, 0.4822, 0.4465], 
        #    std=[0.2470, 0.2435, 0.2616]
        #)
    ]) 

    cifar10 = CIFAR10(dir, True, tforms, download=downloadIfNotExists)
    return get_train_test_split(cifar10, train_test_split, bsize, shuffle, _N)



class _norm_cl(nn.Module):
    def __init__(self, classifier, norm):
        super().__init__()
        if norm is None: norm = nn.Identity()
        self.cl, self.norm = classifier, norm
    def forward(self, x):
        return self.cl(self.norm(x))

def get_normalized_classifier(classifier, norm):
    return _norm_cl(classifier, norm)




def _get_rand_states():
    return \
        np.random.get_state(), \
        torch.get_rng_state(), \
        random.getstate(), \
        (torch.cuda.get_rng_state_all() \
        if torch.cuda.is_available() else None)


def _set_rand_states(data):
    _np, _torch, _base, _cuda = data
    np.random.set_state(_np)
    torch.random.set_rng_state(_torch)
    random.setstate(_base)
    if _cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(_cuda)





class Timer:
    def __init__(self):
        self.start=now()
    def get(self):
        return now()-self.start
    def reset(self):
        res = now()-self.start
        self.start=now()
        return res










def quantize(x:T, depth=32):
    _d = depth #- 1.
    return (x * _d).round() / _d


class _BPDA_quantize(torch.autograd.Function):
    @staticmethod
    def forward(_, x, depth=32):
        return quantize(x, depth)

    @staticmethod
    def backward(_, grad_output):
        # Identity approximation: pass gradient straight through
        return grad_output, None

def _quantize(x:T, _depth=32, _do_bpda=False):
    return _BPDA_quantize.apply(x, _depth) if _do_bpda else quantize(x, _depth)
    



class _BPDA_predict_and_denoise(torch.autograd.Function):
    @staticmethod
    def forward(context, x, t, _non_cont, *args, **kwargs):
        # with torch.no_grad():
        out = _non_cont(x, t, *args, **kwargs).sample
        return out
    @staticmethod
    def backward(_, grad):
        return grad, None,None



# for easier debug
class _BPDA_identity(torch.autograd.Function):
    @staticmethod
    def forward(context, x, t, _non_cont, *args, **kwargs):
        # with torch.no_grad():
        out = _non_cont(x, t, *args, **kwargs).sample
        return out
    @staticmethod
    def backward(_, grad):
        return grad, None,None

# # for the general case
# class _BPDA_arbitrary_identity(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, _non_cont, *args, **kwargs):
#         ctx.num_args = len(args)
#         # with torch.no_grad():
#         out = _non_cont(*args, **kwargs)
#         return out
#     @staticmethod
#     def backward(ctx, *grads):
#         approx_grad = grads[0]
#         return (None, approx_grad, *(None for _ in range(ctx.num_args-1)))



class _BPDA_arbitrary_identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _non_cont, *args, **kwargs):
        ctx.tensor_mask = [torch.is_tensor(a) and a.is_floating_point() for a in args]
        ctx.num_args = len(args)
        with torch.no_grad():
            out = _non_cont(*args, **kwargs)
        return out

    @staticmethod
    def backward(ctx, *grads):
        approx_grad = grads[0]
        # identity-pass grad to every float tensor arg, None to the rest
        arg_grads = [approx_grad if is_t else None for is_t in ctx.tensor_mask]
        return (None, *arg_grads)



def _extract_x(out):
    if isinstance(out, T): 
        return out
    else:
        return _extract_x(out[0])

# for the general case
class _BPDA_arbitrary_SCORE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _non_cont, *args, **kwargs):
        ctx.num_args = len(args)
        x = args[0]
        with torch.no_grad():
            out = _non_cont(*args, **kwargs)
        ctx.save_for_backward(x, _extract_x(out))
        return out

    @staticmethod
    def backward(ctx, *grads):
        # return (None, grad) + (None,) * (ctx.num_args - 1)
        grad = grads[0]
        x, out = ctx.saved_tensors
        residual = (out - x).detach()                          # shape same as x
        sensitivity = 1.0 + residual.abs()            # in [0,1], elementwise
        approx_grad = grad * sensitivity / sensitivity.mean()
        return (None, approx_grad, *(None for _ in range(ctx.num_args-1)))

def make_BPDA_score(diffusion):
    def func(*args, **kwargs):
        return _BPDA_arbitrary_SCORE.apply(diffusion, *args, **kwargs)
    return func

def make_arbitrary_BPDA_identity(_non_cont):
    def func(*args, **kwargs):
        return _BPDA_arbitrary_identity.apply(_non_cont, *args, **kwargs)
    return func

# DDIM energy gradient approximation
class _BPDA_ddim(torch.autograd.Function):
    @staticmethod
    def forward(context, x, t, ddpm, sched, eta):
        with torch.no_grad():
            out = ddpm(x, t)
        context.sched = sched
        context.eta = eta
        context.save_for_backward(x, t)
        return out.sample

    @staticmethod
    def backward(ctx, grad):
        sched = ctx.sched
        x, t = ctx.saved_tensors
        d = x.device

        t_prev = (t - 1).clamp(min=0)
        at = sched.alphas_cumprod[t].to(d)
        atm1 = sched.alphas_cumprod[t_prev].to(d)
        atm1 = torch.where(t == 0, torch.ones_like(atm1), atm1)

        # Deterministic component: sqrt(alpha_{t-1} / alpha_t)
        K_det = torch.sqrt(atm1 / at)

        # Stochastic correction for eta > 0
        if ctx.eta > 0:
            beta_t = 1 - at
            beta_t_prev = 1 - atm1
            variance = (beta_t_prev / beta_t) * (1 - at / atm1)
            sigma = ctx.eta * torch.sqrt(variance.clamp(min=0))
            # Attenuate gradient by the fraction of signal vs noise
            K = K_det * torch.sqrt((1 - sigma**2 / beta_t_prev).clamp(min=0))
        else:
            K = K_det

        K = K.view(-1, 1, 1, 1)
        return K * grad, None, None, None, None