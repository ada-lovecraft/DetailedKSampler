import sys
import os
import torch
from functools import lru_cache
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
import comfy.samplers
import comfy.utils

@lru_cache(maxsize=1)
def remap_range(value, minIn, MaxIn, minOut, maxOut):
    return ((min(max(value, minIn), MaxIn) - minIn) / (MaxIn - minIn)) * (maxOut - minOut) + minOut

def freeu(model, b1, b2, s1, s2):
    m = model.clone()
    del model
    def Fourier_filter(x):
        x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
        x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
        B, C, H, W = x_freq.shape
        crow, ccol = H // 2, W //2
        mask = torch.ones((B, C, H, W), device=x.device)
        mask[..., crow - 1:crow + 1, ccol - 1:ccol + 1] = 1.0
        x_freq *= mask
        x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
        x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real
        return x_filtered.to(x.dtype)
    scale_dict = {m.model.model_config.unet_config["model_channels"] * 4: (b1, s1), m.model.model_config.unet_config["model_channels"] * 2: (b2, s2)}
    on_cpu_devices = {}
    def output_block_patch(h, hsp, transformer_options):
        if scale_dict.get(h.shape[1], None) is not None:
            hidden_mean = h.mean(1).unsqueeze(1)
            hidden_max, _ = torch.max(hidden_mean.view(hidden_mean.shape[0], -1), dim=-1, keepdim=True)
            hidden_min, _ = torch.min(hidden_mean.view(hidden_mean.shape[0], -1), dim=-1, keepdim=True)
            hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
            h[:,:h.shape[1] // 2] = h[:,:h.shape[1] // 2] * ((scale_dict.get(h.shape[1], None)[0] - 1) * hidden_mean + 1)
            if hsp.device not in on_cpu_devices:
                try:
                    hsp = Fourier_filter(hsp)
                except:
                    print("Device", hsp.device, "does not support the torch.fft functions used in the FreeU node, switching to CPU.")
                    on_cpu_devices[hsp.device] = True
                    hsp = Fourier_filter(hsp.cpu()).to(hsp.device)
            else:
                hsp = Fourier_filter(hsp.cpu()).to(hsp.device)
        return h, hsp
    m.set_model_output_block_patch(output_block_patch)
    return m

def patch(model, mimic_scale, threshold_percentile):
    m = model.clone()
    del model
    def sampler_dyn_thrash(args):
        input_args = args["input"]
        cond = input_args - args["cond"]
        uncond = input_args - args["uncond"]
        cond_scale = args["cond_scale"]
        diff = ((cond.reshape((-1, int(cond.shape[0] / uncond.shape[0])) + uncond.shape[1:]) - uncond.unsqueeze(1)).sum(1) if cond.shape[0] % uncond.shape[0] == 0 else print("Error: Expected # of conds per batch to be constant across batches"))
        mim_target = uncond + diff * mimic_scale
        cfg_flattened = (uncond + diff * cond_scale).flatten(2)
        cfg_centered = cfg_flattened - cfg_flattened.mean(dim=2).unsqueeze(2)
        mim_scaleref = (mim_target.flatten(2) - mim_target.flatten(2).mean(dim=2).unsqueeze(2)).abs().max()
        max_scaleref = torch.maximum(mim_scaleref, torch.quantile(cfg_centered.abs(), threshold_percentile))
        return input_args - ((cfg_centered.clamp(-max_scaleref, max_scaleref) / max_scaleref) * mim_scaleref + cfg_flattened.mean(dim=2).unsqueeze(2)).unflatten(2, (uncond + diff * mimic_scale).shape[2:])
    m.set_model_sampler_cfg_function(sampler_dyn_thrash)
    return m

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, start_at_step, end_at_step):
    import latent_preview
    import comfy.sample
    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(latent["samples"], seed, batch_inds)
    noise_mask = latent["noise_mask"] if "noise_mask" in latent else None
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    latent["samples"] = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent["samples"], denoise=1.0, disable_noise=False, start_step=start_at_step, last_step=end_at_step, force_full_denoise=False, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    return latent

def add_latents(samples1, blend_factor, samples2):
    samples1["samples"] = torch.add(torch.mul(samples1["samples"], (1 - blend_factor)), torch.mul(samples2["samples"], blend_factor))
    return samples1

class DetailedKSampler:
    @classmethod
    def INPUT_TYPES(s):
        ui_widgets = {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 23, "min": 0, "max": 10000}),
                "extra_steps": ("INT", {"default": 5, "min": 0, "max": 10000}),
                "missing_steps": ("INT", {"default": 92, "min": 0, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_3m_sde"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_level": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 2.0, "step": 0.1}),
                "detail_from": (["penultimate_step", "sample"], {"default": "penultimate_step"}),
                "auto_threshold": (["enable", "disable"], {"default": "enable"}),
                "mimic_scale": ("FLOAT", {"default": 9.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "threshold_percentile": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "auto_freeu": (["enable", "disable"], {"default": "enable"}),
                "b1": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 4.0, "step": 0.01}),
                "b2": ("FLOAT", {"default": 1.4, "min": 0.25, "max": 2.1, "step": 0.01}),
                "s1": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.4, "step": 0.01}),
                "s2": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.4, "step": 0.01}),
                "copilot": (["enable", "disable"], {"default": "disable"}),
            },
            "optional": {
                "second_model": ("MODEL",),
                "VAE": ("VAE",),
            },
        }
        return ui_widgets

    RETURN_TYPES = ("LATENT","IMAGE",)
    FUNCTION = "detailed_sample"
    CATEGORY = "sampling"

    def detailed_sample(self, model, seed, steps, extra_steps, missing_steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, detail_level, detail_from, auto_threshold, mimic_scale, threshold_percentile, auto_freeu, b1, b2, s1, s2, copilot, second_model=None, VAE=None):
        if sampler_name in ["uni_pc", "uni_pc_bh2"]:
            steps = steps - 1
            sampler_name = "dpmpp_3m_sde"
            scheduler = "karras"
        if copilot == "enable":
            extra_steps = round(steps * 0.22)
            missing_steps = steps * 5 - steps
            mimic_scale = max(0, min(13 - cfg, 12) - (min(13 - cfg, 12) % 0.5))
            print("extra_steps:", extra_steps, ", missing_steps:", missing_steps, ", mimic_scale:", mimic_scale)
        total_steps = steps + missing_steps + extra_steps
        if denoise < 1.0:
            denoise_steps = round(total_steps / denoise)
            start_at_step = denoise_steps - total_steps
            total_steps = denoise_steps
            steps = start_at_step + steps
        else:
            start_at_step = 0
        start_at_extra_step = steps + missing_steps
        if auto_freeu == "enable":
            model = freeu(model,b1,b2,s1,s2)
        if auto_threshold == "enable":
            model = patch(model,mimic_scale,threshold_percentile)
        sample_model = common_ksampler(model,seed,steps,cfg,sampler_name,scheduler,positive,negative,latent_image,start_at_step,steps)
        if second_model is not None:
            if auto_freeu == "enable":
                second_model = freeu(second_model,b1,b2,s1,s2)
            if auto_threshold == "enable":
                second_model = patch(second_model,mimic_scale,threshold_percentile)
            extram = second_model
        else:
            extram = model
        if detail_level != 1.0:
            if detail_from == "penultimate_step":
                noisy_latent_1 = common_ksampler(extram,seed,steps+1,cfg,sampler_name,scheduler,positive,negative,latent_image,steps,steps+1) if detail_level > 1.0 else common_ksampler(model,seed,steps,cfg,sampler_name,scheduler,positive,negative,latent_image,steps-1,steps)
            else:
                noisy_latent_1 = sample_model
            noisy_latent_2 = common_ksampler(extram,seed,total_steps,cfg,sampler_name,scheduler,positive,negative,latent_image,start_at_extra_step,total_steps)
            sample_output = add_latents(noisy_latent_1,remap_range(detail_level,1.0,2.0,1.0,0.0),noisy_latent_2) if detail_level > 1.0 else add_latents(noisy_latent_1,detail_level,noisy_latent_2)
        else:
            sample_output = common_ksampler(extram,seed,total_steps,cfg,sampler_name,scheduler,positive,negative,latent_image,start_at_extra_step,total_steps)
        return (sample_output, VAE.decode(sample_output["samples"]),) if VAE is not None else (sample_output, torch.cat((torch.full([1, 512, 512, 1], 0), torch.full([1, 512, 512, 1], 0), torch.full([1, 512, 512, 1], 0)), dim=-1),)

class DetailedKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        ui_widgets = {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 23, "min": 0, "max": 10000}),
                "extra_steps": ("INT", {"default": 5, "min": 0, "max": 10000}),
                "missing_steps": ("INT", {"default": 92, "min": 0, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_3m_sde"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "detail_level": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 2.0, "step": 0.1}),
                "detail_from": (["penultimate_step", "sample"], {"default": "penultimate_step"}),
                "auto_threshold": (["enable", "disable"], {"default": "enable"}),
                "mimic_scale": ("FLOAT", {"default": 9.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "threshold_percentile": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "auto_freeu": (["enable", "disable"], {"default": "enable"}),
                "b1": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 4.0, "step": 0.01}),
                "b2": ("FLOAT", {"default": 1.4, "min": 0.25, "max": 2.1, "step": 0.01}),
                "s1": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.4, "step": 0.01}),
                "s2": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.4, "step": 0.01}),
                "copilot": (["enable", "disable"], {"default": "disable"}),
            },
            "optional": {
                "second_model": ("MODEL",),
                "VAE": ("VAE",),
            },
        }
        return ui_widgets

    RETURN_TYPES = ("LATENT","IMAGE",)
    FUNCTION = "detailed_sample_advanced"
    CATEGORY = "sampling"

    def detailed_sample_advanced(self, model, seed, steps, extra_steps, missing_steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, detail_level, detail_from, auto_threshold, mimic_scale, threshold_percentile, auto_freeu, b1, b2, s1, s2, copilot, second_model=None, VAE=None):
        if sampler_name in ["uni_pc", "uni_pc_bh2"]:
            steps = steps - 1
            sampler_name = "dpmpp_3m_sde"
            scheduler = "karras"
        if copilot == "enable":
            extra_steps = round(steps * 0.22)
            missing_steps = steps * 5 - steps
            mimic_scale = max(0, min(13 - cfg, 12) - (min(13 - cfg, 12) % 0.5))
            print("extra_steps:", extra_steps, ", missing_steps:", missing_steps, ", mimic_scale:", mimic_scale)
        total_steps = steps + missing_steps + extra_steps
        start_at_extra_step = steps + missing_steps
        if auto_freeu == "enable":
            model = freeu(model,b1,b2,s1,s2)
        if auto_threshold == "enable":
            model = patch(model,mimic_scale,threshold_percentile)
        sample_model = common_ksampler(model,seed,steps,cfg,sampler_name,scheduler,positive,negative,latent_image,start_at_step,steps)
        if second_model is not None:
            if auto_freeu == "enable":
                second_model = freeu(second_model,b1,b2,s1,s2)
            if auto_threshold == "enable":
                second_model = patch(second_model,mimic_scale,threshold_percentile)
            extram = second_model
        else:
            extram = model
        if detail_level != 1.0:
            if detail_from == "penultimate_step":
                noisy_latent_1 = common_ksampler(extram,seed,steps+1,cfg,sampler_name,scheduler,positive,negative,latent_image,steps,steps+1) if detail_level > 1.0 else common_ksampler(model,seed,steps,cfg,sampler_name,scheduler,positive,negative,latent_image,steps-1,steps)
            else:
                noisy_latent_1 = sample_model
            noisy_latent_2 = common_ksampler(extram,seed,total_steps,cfg,sampler_name,scheduler,positive,negative,latent_image,start_at_extra_step,total_steps)
            sample_output = add_latents(noisy_latent_1,remap_range(detail_level,1.0,2.0,1.0,0.0),noisy_latent_2) if detail_level > 1.0 else add_latents(noisy_latent_1,detail_level,noisy_latent_2)
        else:
            sample_output = common_ksampler(extram,seed,total_steps,cfg,sampler_name,scheduler,positive,negative,latent_image,start_at_extra_step,total_steps)
        return (sample_output, VAE.decode(sample_output["samples"]),) if VAE is not None else (sample_output, torch.cat((torch.full([1, 512, 512, 1], 0), torch.full([1, 512, 512, 1], 0), torch.full([1, 512, 512, 1], 0)), dim=-1),)

NODE_CLASS_MAPPINGS = {
    "DetailedKSampler": DetailedKSampler,
    "DetailedKSamplerAdvanced": DetailedKSamplerAdvanced
}
