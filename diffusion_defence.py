import torch
from torch import nn, Tensor as T
from torch.utils.checkpoint import checkpoint
import diffusers
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import compute_snr

from torch_common_clean import *
from torch_common_clean import _BPDA_ddim, _BPDA_identity
from torch_common_clean import make_arbitrary_BPDA_identity, make_BPDA_score
from torch_common_clean import _quantize

from models import sched_conf_path

from guided_scheduler import DDIMScheduler_guided

from frequency_band_blending import _FFT_reconstruct_from_two_circle












# simple ddpm loop used to purify as described in 
# https://arxiv.org/abs/2205.07460
# using DDIM scheduler
# https://arxiv.org/abs/2010.02502
class DiffPure(nn.Module):
    def __init__(self, ddpm, classifier, 
                 steps = 10, noise_add = 100,
                 classifier_norm_function = None, 
                 sched_conf_path=None,
                 T = 1_000,
                 _bpda=BPDA_TYPE):
        super().__init__()

        # transforms to ensure data is in the correct format 
        # for diffusion and classifier models
        classifier_norm_function = (lambda x : x) \
            if classifier_norm_function is None else classifier_norm_function 
        self.norm_f = lambda x : x * 2 - 1
        self.norm_f_inv = lambda x : classifier_norm_function(((x + 1.) / 2.).clamp(0,1))

        self.ddpm=ddpm
        self.classifier = classifier

        # initialize schedule, and set timesteps according to noise_add
        # DDIM scheduler used, by Song et al:
        # https://arxiv.org/abs/2010.02502
        self.sched = DDIMScheduler.from_config(sched_conf_path) \
                     if sched_conf_path is not None else        \
                     DDIMScheduler(T, 1e-6, 2e-2)


        # scaling to ensure purification magnitude scales with 
        # amount of steps and noise added, same as Li et al:
        # https://github.com/LixiaoTHU/ADBM/blob/main/defensediff.py
        # self.sched.set_timesteps(1000//(noise_add//steps))
        self.sched.set_timesteps(1000//(max(5,noise_add//steps)))
        # self.sched.set_timesteps(1000)


        self.do_noise_add_ = noise_add != 0
        self.noise_add = noise_add
        self.steps =steps
        self.T=T
        self.bpda = _bpda

        # DDIM stochasticity 
        self._eta = .3

        self.training = False

        self.ddpm_call = make_arbitrary_BPDA_identity(
                 lambda x, t : self.ddpm(x, t).sample) if self.bpda in ["full", "ddpm"]\
            else lambda x, t : self.ddpm(x, t).sample

        # self.training = self

    # only get x(t-1) given x(t), slightly denoise x
    def _get_prev_checkpoint(self, x, t, device):
        ts = t.expand(x.shape[0]).to(device)
        noise_pred = self.ddpm_call(x, ts)
        out = self.sched.step(noise_pred, t, x)
        return out.prev_sample

    # checkpointing to allow large diffusion chains to be held in memory
    def _get_prev(self, x, t, device):
        return self._get_prev_checkpoint(x, t, device) \
               if (not LOW_VRAM) else \
               checkpoint(self._get_prev_checkpoint, x, t, device,
                          use_reentrant=False)


    def set_steps(self, steps):
        self.steps = steps
        self.sched.set_timesteps(1000//(max(5,self.noise_add//steps)))

    def forward(self, x):

        # x = self.norm_f(x)#.clone()


        if self.bpda in ["full", "score", "diffpure"]:
            _purify = make_arbitrary_BPDA_identity(self._forward)
        else:
            _purify = self._forward

        purified = _purify(x)#.sample
        # scale and classify 
        out = self.classifier(purified)

        return out

    def _forward(self, x):
        return self._forward_checkpoint(x) \
               if (not LOW_VRAM) else \
               checkpoint(self._forward_checkpoint, x,
                          use_reentrant=False)

    def _forward_checkpoint(self, x):
        device = x.device
        x = self.norm_f(x)
        # add noise at the start to attempt drowning out peturbations
        if self.do_noise_add_:
            noise = torch.randn_like(x)
            prev = self.sched.add_noise(
                    x, noise, 
                    T([self.noise_add])
                        .long()
                            .to(device))
        else:
            prev = x

        # perform ddpm loop
        for t in self.sched.timesteps[-self.steps:]:
            prev = self._get_prev(prev, t, device)
            # prev = self._get_prev(prev, t.clamp(max=MAX_t), device)


        return self.norm_f_inv(prev)







# DiffPure but with x0 estimate majority voting
class DiffPure_x0v(DiffPure):
    def __init__(self, ddpm, classifier, 
                 steps = 10, noise_add = 100,
                 classifier_norm_function = None, 
                 sched_conf_path=sched_conf_path,
                 T = 1_000,
                 _bpda=BPDA_TYPE):
        super().__init__(ddpm, classifier, steps, noise_add,
                         classifier_norm_function, sched_conf_path, T,
                         _bpda)

    # TODO SNR
    # given output logits of the current x0 prediction, 
    # scale it such that later predictions have higher weight
    def _scale_vote(self, logits, i, n, t):
        # for all of these, the first term scales logits
        # the second term normalizes such that sum(scales)=1
        # K = (i/n) * 2./(n+1.)
        snr = compute_snr(self.sched, t)
        K = snr * 1e-3
        # snr_norm = snr / (1. + snr)
        # print(i, t, snr)
        #K = ((i/n)**2) * (6.*n / (2*n*n + 3*n + 1))
        #K = ((i/n)**3) * (4*n) / ((n+1.)**2)
        out = torch.exp(logits) * K
        return out


    # def _predict_noise_and_purify_backend(self, x, t, i, device):
    #     ts = t.expand(x.shape[0]).to(device)
    #     noise_pred = self.ddpm(x, ts).sample
    #     out = self.sched.step(noise_pred, t, x)
    #     return out.prev_sample, out.pred_original_sample

    # def _predict_and_purify(self, x, t, i, device):
    #     if self.bpda == "full":
    #         func = 


    # given an adversarial datapoint, predict x0 (non adversarial datapoint)
    # save scaled logits, and return slightly less noisey image for next iteration
    def _get_pre_vote_checkpoint(self, x, t, i, device, return_x0_pred):

        # format data and predict noise (out of distribution data)
        ts = t.expand(x.shape[0]).to(device)
        # if self.bpda is None:
        #     noise_pred = self.ddpm(x, ts).sample
        # elif self.bpda == "full":
        #     noise_pred = _BPDA_identity.apply(x, ts, self.ddpm)
        noise_pred = self.ddpm_call(x, ts)

        # perform one denoising step using DDIM sampler 
        out = self.sched.step(noise_pred, t, x)
        prev = out.prev_sample#.clone()
        
        # gotten for free using ddim sampler
        orig_pred = out.pred_original_sample

        # self.t.append((((orig_pred+1.)/2.).clamp(0,1)))

        orig_pred = self.norm_f_inv(orig_pred)


        # prev = orig_pred

        return (prev, orig_pred) if (not return_x0_pred) else ((prev, orig_pred), orig_pred)


    def classify(self, pre_vote, i, t):
        logits = self.classifier(pre_vote)
        vote = self._scale_vote(logits, i, self.steps, t)#.clone()
        return vote

    # low vram call of the above method
    def get_pre_vote(self, x, t, i, device=gpu, return_x0_pred=False):

        vote_function = make_arbitrary_BPDA_identity(self._get_pre_vote_checkpoint) \
            if (self.bpda in ["full", "vote", "ddpm"]) else self._get_pre_vote_checkpoint\
            

        return vote_function(x,t,i,device, return_x0_pred) \
               if (not LOW_VRAM) else \
               checkpoint(vote_function, x,t,i,device,return_x0_pred,
                          use_reentrant=False)


    def forward(self, x):
        shape = x.shape
        # print(shape); exit()

        # same as diffpure
        device=x.device
        x = self.norm_f(x).clone()
        if self.do_noise_add_:
            noise = torch.randn_like(x)
            prev = self.sched.add_noise(
                    x, noise, 
                    T([self.noise_add])
                        .long()
                            .to(device))
        else:
            prev = x

        out = torch.zeros_like(
            self.classifier(
                torch.zeros(shape).to(device)))

        for i, t in en(self.sched.timesteps[-self.steps:]):
            prev, out = checkpoint(self._one_step, prev, t, i, device, out,
                                   use_reentrant=False)
        # topil(torch.cat(self.t)).save("X0S.png")

        return out

    def _one_step(self, prev, t, i, device, out):
        prev, pre_vote = self.get_pre_vote(prev, t.clamp(max=MAX_t), i, device)
        vote = self.classify(pre_vote, i, t)
        return prev, out + vote / self.steps


    def set_steps(self, steps):
        self.steps = steps
        self.sched.set_timesteps(1000//(max(5,self.noise_add//steps)))


# for boolean control over torch.no_grad ect
class IdentityContextManager:
    def __init__(self):
        pass;
    def __enter__(self, *_, **__):
        return self
    def __exit__(self, *_, **__):
        pass



class DiffPure_guided(DiffPure_x0v):
    def __init__(self, ddpm, classifier, 
                 steps = 100, noise_add = 100,
                 classifier_norm_function = None, 
                 sched_conf_path=sched_conf_path,
                 peturb_eps=8/255,
                 guidance_scale=1.,
                 T = 1_000,
                 _quant_level = 16,
                 _target="fft",
                 _vote_replace_type="fft",
                 _avg_over_n=5,
                 _bpda=BPDA_TYPE,
                 _super_stochastic=False,
                 _reverse_freq=False,
                 _scale_by_snr=False,
                 _low_freq_func="q",
                 _replace_x_with_x_guide_per_step=False,
                 _is_freqpure=False,
                 _fft_r=.1,
                 _disable_freq=False,
                 _without=None):

        steps = steps // _avg_over_n

        super().__init__(ddpm, classifier, steps, noise_add,
                         classifier_norm_function, sched_conf_path, T)
        

        self._vote_replace_type = _vote_replace_type
        self.eps=peturb_eps
        # self.sched.__class__ = DDIMScheduler_guided
        self.sched = DDIMScheduler_guided.from_config(self.sched.config)
        # self.sched.set_timesteps(1000)
        self.sched.set_timesteps(1000//(max(5,noise_add//steps)))

        self.fft_r = _fft_r
        self._current_adv = None
        self.gscale = guidance_scale
        self._scale_by_snr=_scale_by_snr
        self._replace_per_step = _replace_x_with_x_guide_per_step

        self._no_freq= _disable_freq

        self._super_stochastic = _super_stochastic
        self._reverse_freq = _reverse_freq

        self.guidance_target = None
        self.bpda = _bpda

        self._is_freqpure = _is_freqpure

        if self._is_freqpure:
            self.gscale =0

        if self._is_freqpure or _target == "adv":
            self._low_freq_func = None;


        if _low_freq_func=="q":
            self._low_freq_func = lambda x : (_quantize(
                x, _quant_level, self.bpda is not None))
            # self._low_freq_func = lambda x : self._lowpass_fft(_quantize(
            #     x, _quant_level, self.bpda is not None), .01)
        elif _low_freq_func is None:
            self._low_freq_func = lambda x : x
        elif self._low_freq_func == "fft":
            self._low_freq_func = self._lowpass_fft
        else:
            raise Exception("bad low freq func argument")



        # print(self._low_freq_func)

        self._target_adv=_target
        # self.x0s= []
        # self.fextract._optimize_fft_k()

        # self._rec_fft_r = .5
        self.training = False
        self._cache_noise=None
        self._randomize_noise=True
        self._eta = .3
        self.wo = _without
        self._avg_over_N = _avg_over_n




    def set_steps(self, steps):
        self.steps = steps
        self.sched.set_timesteps(1000//(max(5,self.noise_add//steps)))

    def _condition(self, *args, **kwargs):
        _cond = make_arbitrary_BPDA_identity(
           self._condition_checkpoint) if self.bpda in ["full", "cond"] else \
           self._condition_checkpoint

        return _cond(*args, **kwargs)



    # def _recompute_x0(self, t, x_in, x0, device):
    #     alpha_prod_t = self.sched.alphas_cumprod[t].to(device)
    #     beta_prod_t = 1 - alpha_prod_t
    #     with torch.no_grad(): # required no grad for consistency
    #         eps_fixed = (x_in - alpha_prod_t**0.5 * x0) / beta_prod_t**0.5
    #     return (x_in - beta_prod_t**0.5 * eps_fixed) / alpha_prod_t**0.5

    def _condition_checkpoint(self, x_in:T, x_adv:T, t, x0, x_guide)->T:

        if self.gscale==0: return None

        # ensure gradients are computed through this part if not
        # using bpda on this part
        # ctx_manager = IdentityContextManager    \
        #     if self.bpda in ["full", "cond"] else\
        #     torch.enable_grad
        # ctx_manager = IdentityContextManager()

        # with ctx_manager():

        device=x_in.device
        x_adv = x_adv.to(device)
        # x0 = x0.to(device).requires_grad_(True)

        if self._target_adv == "adv":
            noise = torch.randn_like(x_adv)
            noised = self.sched.add_noise(
                    x_adv, noise, 
                    T([self.noise_add])
                        .long()
                            .to(device))
            target = noised
        elif self._target_adv == "fft":
            target = x_guide
        else:
            raise Exception()

        # target = x_guide
        _eps = 1
        _at = self.sched.alphas_cumprod[t].to(device)

        # scale as Wang et al, 
        # https://github.com/JinyiW/GuidedDiffusionPur
        scale = self.gscale * (3 * torch.sqrt(1 - _at)) / ((_eps * torch.sqrt(_at)) + EPSILON)
        scale = scale.clamp(max=1e4)

        # analytic gradient to avoid storing second derivative tensors in vram
        grad_x0 = (target - x0)
        grad_xt = grad_x0 / (_at ** 0.5).clamp(min=EPSILON)
        norm = grad_xt.reshape(grad_xt.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1)
        grad_xt = grad_xt / norm.clamp(min=EPSILON)
        out = (grad_xt * scale)#.clone()



        if torch.isnan(out).any():
            print(f"NaN in condition at t={t}, scale={scale}, norm={norm.min()}, grad_x0_mean={grad_x0.abs().mean()}, guidance={self.gscale}, x0={x0.mean()}, ff = {self._low_freq_func}")
            return None  # fail safe

        return out#.clone()#.requires_grad_(False)




    def get_pre_vote(self, x, t, i, x_guide, device=gpu, return_x0_pred=False):

        vote_function = self._get_pre_vote_checkpoint if self.bpda not in ["full", "vote"] else\
             make_arbitrary_BPDA_identity(self._get_pre_vote_checkpoint)

        return vote_function(x,t,i,x_guide,device, return_x0_pred) \
               if (not LOW_VRAM) else \
               checkpoint(vote_function, x,t,i,x_guide,device,return_x0_pred,
                          use_reentrant=False)





    def _forward(self, x):
        shape = x.shape
        # print(shape); exit()

        # same as diffpure
        device=x.device
        x = self.norm_f(x).clone()
        if self.do_noise_add_:
            noise = torch.randn_like(x)
            prev = self.sched.add_noise(
                    x, noise, 
                    T([self.noise_add])
                        .long()
                            .to(device))
        else:
            prev = x

        x_adv = x

        out = torch.zeros_like(
            self.classifier(
                torch.zeros(shape).to(device)))

        if self._super_stochastic:
            _n = self.steps + torch.randint(-self.steps // 3, self.steps // 3 + 1, (1,)).item()
            _n = max(3, _n)
        else:
            _n = self.steps

        # ts = T([MAX_t]).long().expand(x.shape[0]).to(device)
        # noise_pred = self.ddpm(x, ts).sample
        # _out = self.sched.step(noise_pred, T([MAX_t]).long(), x,
        #                       _condition = None,
        #                       x_adv=self._current_adv,
        #                       eta=self._eta)
        
        # # gotten for free using ddim sampler
        # x0 = _out.pred_original_sample
        x_guide = x
        x0 = None

        

        for i, t in (en(self.sched.timesteps[-_n:])):
            prev, vote, x0, x_guide = checkpoint(self._step,
                                                 x0, x_guide, i, t,
                                                 x_adv, prev, device,
                                                 use_reentrant=False)
            out = out + vote / self.steps

        # topil(torch.cat(self.t)).save("X0S.png")


        if self._target_adv == "adv" or self._is_freqpure:
            out = self.norm_f_inv(prev)
            out = self.classifier(out)

        return out


    def _step(self, x0, x_guide, i, t, x_adv, prev, device):
        # voting

        is_vanilla = self._target_adv != "adv" and (not self._is_freqpure)
        if self._super_stochastic:
            t = (t * int(1 + torch.randint(1, 10, (1,))[0] / 20 - .25)).int()
            
        _FFT_func = _FFT_reconstruct_from_two_circle if (not self._no_freq) else\
                    lambda a, b, t : a

        if x0 is not None:
            if is_vanilla:
                A=self._low_freq_func(x_adv).clamp(0, 1)
                A=mask_nans(A, x_adv)
                B=x0
                x_guide = _FFT_func(
                    (A+1.)/2., (B+1.)/2., self.fft_r) * 2. - 1.
            else:
                x_guide = x_adv


        (prev, x0), pre_vote = self.get_pre_vote(prev, t.clamp(max=MAX_t), i, 
                                            x_guide, device, True)
        vote = self._classify(pre_vote, i, t)
        

        return prev, vote, x0, x_guide





    def forward(self, x, *args, **kwargs):

        self._current_adv = self.norm_f(x).clone().to(x.device)
        self._cache_noise = torch.randn_like(x).to(x.device)

        outs = []

        for _ in range(self._avg_over_N):
            self._cache_noise = torch.randn_like(x).to(x.device)
            # outs.append(super().forward(x, *args, **kwargs))
            outs.append(self._forward(x, *args, **kwargs))

        return torch.stack(outs).mean(0)


    # def _get_logits


    def _classify(self, pre_vote, i, t):

        logits = self.classifier(
            self.norm_f_inv(pre_vote))

        vote = self._scale_vote(logits, i, self.steps, t).clone()


        if self._super_stochastic:
            if torch.rand(1).item() < .5:
                vote = torch.zeros_like(vote)
            prev = _quantize(prev, randint(8, 32), self.bpda is not None)

        return vote



    def _get_pre_vote_checkpoint(self, x, t, i, x_guide, device, return_x0_pred):

        assert self._current_adv is not None, "bad internal state"

        if x_guide is None: 
            x_adv = self._current_adv
            A=self._low_freq_func(x_adv).clamp(0, 1)
            x_guide = A
            # A=mask_nans(A, x_adv)
            # B=x0
            # x_guide = _FFT_reconstruct_from_two_circle(
            #     (A+1.)/2., (B+1.)/2., self.fft_r) * 2. - 1.

        # regular ddpm loop process-
        ts = t.expand(x.shape[0]).to(device)

        noise_pred = self.ddpm_call(x, ts)

        x_t =  self.sched.add_noise(
                    x, torch.randn_like(x), 
                    T([10])
                        .long()
                            .to(device))
            
        _condition = lambda x_in, x_adv, t, x0 : \
            self._condition(x_in, x_adv, t, x0, x_guide)

        out = self.sched.step(noise_pred, t, x,
                              _condition = _condition,
                              x_adv=self._current_adv,
                              eta=self._eta)
        prev = out.prev_sample.clone()
        
        # gotten for free using ddim sampler0
        orig_pred = out.pred_original_sample

        _purified = x_guide

        if self._replace_per_step:
            prev = _purified
     
   

        prev = torch.nan_to_num(prev, nan=0.0).clamp(-3.0, 3.0)

        return (prev, _purified) if (not return_x0_pred) else ((prev, orig_pred), _purified)



    def _lowpass_fft(self, x, r=None):
        # x in [-1, 1] or [0, 1] � operate in whichever space, just be consistent
        r = self.fft_r if r is None else r
        X = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
        H, W = x.shape[-2:]
        cy, cx = H // 2, W // 2
        yy, xx = torch.meshgrid(
            torch.arange(H, device=x.device) - cy,
            torch.arange(W, device=x.device) - cx,
            indexing="ij")
        d = (yy.float()**2 + xx.float()**2).sqrt()
        cutoff = min(1, r * min(H, W) / 2)
        # smooth (Butterworth-like) mask � differentiable, no flat dead zone
        mask = 1.0 / (1.0 + (d / cutoff)**(2*4))
        X = X * mask
        out =  torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1)), norm="ortho").real


        return torch.nan_to_num(out, 0, 0, 0)
