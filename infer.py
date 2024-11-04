import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, '../')))
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '32'

import shutil
import os.path as osp
import argparse
import torch

torch.backends.cuda.matmul.allow_tf32 = True

from functools import partial
from lib.core.webui.shared_opts import send_to_click, set_seed
from lib.core.webui.tab_img_to_3d import create_interface_img_to_3d
from lib.core.webui.tab_3d_to_3d import create_interface_3d_to_3d
from lib.core.webui.tab_text_to_img_to_3d import create_interface_text_to_img_to_3d
from lib.core.webui.tab_retexturing import create_interface_retexturing
from lib.core.webui.tab_3d_to_video import create_interface_3d_to_video
from lib.core.webui.tab_stablessdnerf_to_3d import create_interface_stablessdnerf_to_3d
from lib.apis.adapter3d import Adapter3DRunner
from lib.version import __version__



from lib.core.webui.parameters import (image_defaults, nerf_mesh_defaults, superres_defaults)
from diffusers import (
    EulerAncestralDiscreteScheduler, AutoencoderKL, UNet2DConditionModel, ControlNetModel, StableDiffusionPipeline)

from mmcv.runner import set_random_seed
from lib.pipelines.utils import (
    init_base_modules, init_mvedit, rgba_to_rgb, do_segmentation, do_segmentation_pil, pad_rgba_image, join_prompts,
    zero123plus_postprocess, init_instant3d, init_zero123plus)

import PIL.Image as Image

DEBUG_SAVE_INTERVAL = {
    0: None,
    1: 4,
    2: 1}

def parse_args():
    parser = argparse.ArgumentParser(description='MVEdit 3D Toolbox')
    parser.add_argument('--diff-bs', type=int, default=4, help='Diffusion batch size')
    parser.add_argument('--advanced', action='store_true', help='Show advanced settings')
    parser.add_argument('--debug', choices=[0, 1, 2], type=int, default=0,
                        help='Debug mode - 0: off, 1: on, 2: verbose (visualize everything)')
    parser.add_argument('--local-files-only', action='store_true',
                        help='Only load local model weights and configuration files')
    parser.add_argument('--no-safe', action='store_true', help='Disable safety checker to free VRAM')
    parser.add_argument('--empty-cache', action='store_true', help='Empty the cache directory')
    parser.add_argument('--unload-models', action='store_true', help='Auto-unload unused models to free VRAM')
    parser.add_argument('--share', action='store_true', help='Enable Gradio sharing')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 instead of BF16')
    parser.add_argument('--gs-opacity-thr', type=float, default=0.025, help='GS opacity threshold')
    return parser.parse_args()

# def run_text_to_img(self, seed, *args, **kwargs):
#         image_kwargs = parse_2d_args(list(args), kwargs)
#         self.load_stable_diffusion(image_kwargs['checkpoint'])
#         self.load_scheduler(image_kwargs['checkpoint'], image_kwargs['scheduler'])
#         pipe = StableDiffusionPipeline(
#             vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
#             scheduler=self.scheduler, safety_checker=self.safety_checker, feature_extractor=self.feature_extractor,
#             requires_safety_checker=False)
#         self.unload_ip_adapter(pipe)
#         if self.unload_models:
#             self.unload_stablessdnerf()
#             self.unload_controlnet_ip2p()
#         if self.empty_cache:
#             torch.cuda.empty_cache()

#         print(f'\nRunning text-to-image with seed {seed}...')
#         print(image_kwargs)
#         set_random_seed(seed, deterministic=True)
#         out_img = pipe(
#             height=image_kwargs['height'],
#             width=image_kwargs['width'],
#             prompt=join_prompts(image_kwargs['prompt'], image_kwargs['aux_prompt']),
#             negative_prompt=join_prompts(image_kwargs['negative_prompt'], image_kwargs['aux_negative_prompt']),
#             num_inference_steps=image_kwargs['steps'],
#             guidance_scale=image_kwargs['cfg_scale'],
#             return_dict=False)[0][0]
#         print('Text-to-Image finished.')
#         return out_img

def create_base_opts(var_dict,
                     scheduler='EulerAncestralDiscrete',
                     scheduler_dropdown=['DPMSolverMultistep', 'DPMSolverMultistepKarras',
                                         'DPMSolverSDE', 'DPMSolverSDEKarras',
                                         'EulerAncestralDiscrete', 'DDIM'],
                     steps=24,
                     denoising_strength=0.5,
                     random_init=False,
                     cfg_scale=7,
                     adapter_scale=None,
                     render=True):
    var_dict['scheduler'] = scheduler
    var_dict['steps'] = steps
    if denoising_strength is not None:
        var_dict['denoising_strength'] = denoising_strength
        var_dict['random_init'] = random_init
    if adapter_scale is None:
        var_dict['cfg_scale'] = cfg_scale
    else:
        var_dict['cfg_scale'] =cfg_scale
        var_dict['adapter_scale'] =adapter_scale

def create_prompt_opts(var_dict):
    var_dict['prompt'] = 'a red car'
    var_dict['negative_prompt'] = ''

def text_to_img(sd_api):
    var_dict = dict()
    create_prompt_opts(var_dict)
    create_base_opts(
        var_dict,
        steps=32, denoising_strength=None, cfg_scale=image_defaults['cfg_scale'])
    
    default_var_dict = {
        k: v for k, v in image_defaults.items()
        if k not in var_dict}
    text_to_img_fun = partial(sd_api, **default_var_dict)
    img_to_3d_inputs = {k: var_dict[k] for k in image_defaults.keys()
                        if k not in default_var_dict}
    var_dict['last_seed'] = set_seed(100)
    return text_to_img_fun(var_dict['last_seed'], **img_to_3d_inputs)

def create_superres_opts(
        var_dict,
        superres_defaults,
        do_superres=True, use_ip_adapter=False, scheduler='EulerAncestralDiscrete',
        scheduler_dropdown=['DPMSolverMultistep', 'DPMSolverMultistepKarras',
                            'DPMSolverSDE', 'DPMSolverSDEKarras',
                            'EulerAncestralDiscrete', 'DDIM'],
        steps=24, denoising_strength=0.4, random_init=False,
        n_inverse_steps=48,
        show_advanced=True):
    var_dict['do_superres'] = do_superres
    var_dict['use_ip_adapter'] = use_ip_adapter
    create_base_opts(
        var_dict, scheduler=scheduler, scheduler_dropdown=scheduler_dropdown,
        steps=steps, denoising_strength=denoising_strength,
        random_init=random_init, cfg_scale=superres_defaults['cfg_scale'])

def img_to_3d(
        segmentation_api, zero123plus_api, mvedit_api, api_names=None, advanced=True, init_inverse_steps=640,
        n_inverse_steps=96, diff_bs=6, tet_resolution=128, superres_n_inverse_steps=640, num_passes=6,
        pred_normal=False, image=None):
    default_var_dict = dict(
        init_inverse_steps=init_inverse_steps, n_inverse_steps=n_inverse_steps, diff_bs=diff_bs,
        tet_resolution=tet_resolution)
    default_superres_var_dict = dict(
        n_inverse_steps=superres_n_inverse_steps)

    var_dict = dict()
    create_prompt_opts(var_dict)
    create_base_opts(
        var_dict, scheduler='DPMSolverMultistep',
        steps=24, denoising_strength=0.5, random_init=False,
        cfg_scale=nerf_mesh_defaults['cfg_scale'])
    var_dict['superres'] = dict()
    create_superres_opts(
        var_dict['superres'], superres_defaults,
        use_ip_adapter=True, scheduler='DPMSolverSDEKarras',
        n_inverse_steps=superres_n_inverse_steps, show_advanced=advanced)

    default_var_dict = {
        k: default_var_dict.get(k, v) for k, v in nerf_mesh_defaults.items()
        if k not in var_dict}
    default_superres_var_dict = {
        'superres_' + k: default_superres_var_dict.get(k, v) for k, v in superres_defaults.items()
        if k not in var_dict['superres']}
    img_to_3d_fun = partial(mvedit_api, **default_var_dict, **default_superres_var_dict,
                            cache_dir='cache')
    img_to_3d_inputs =  [var_dict[k] for k in nerf_mesh_defaults.keys()
                        if k not in default_var_dict] + \
                        [var_dict['superres'][k] for k in superres_defaults.keys()
                        if 'superres_' + k not in default_superres_var_dict]
    
    seed = set_seed(100)
    fg_image = segmentation_api(image)
    fg_image.save('outputs/fg_image.png')
    print("Segmentation done")
    var_dict['fg_image'] = fg_image
    var_dict['zero123plus_outputs'] = images = zero123plus_api(seed, fg_image)
    print("Zero123++ done")
    for i, img in enumerate(images):
        img.save(f'outputs/zero123plus_output_{i}.png')
    model_outputs = img_to_3d_fun(seed,images, **img_to_3d_inputs)
    print("3D model generated")

    return model_outputs


def main():
    args = parse_args()

    torch.set_grad_enabled(False)
    runner = Adapter3DRunner(
        device=torch.device('cuda'),
        local_files_only=args.local_files_only,
        unload_models=args.unload_models,
        out_dir=osp.join(osp.dirname(__file__), 'viz') if args.debug > 0 else None,
        save_interval=DEBUG_SAVE_INTERVAL[args.debug],
        save_all_interval=1 if DEBUG_SAVE_INTERVAL[args.debug] == 2 else None,
        dtype=torch.float16 if args.fp16 else torch.bfloat16,
        debug=args.debug > 0,
        no_safe=args.no_safe
    )
    # image = text_to_img(runner.run_text_to_img)
    # print("Image generated")
    # image.save('outputs/output.png')
    image = Image.open('outputs/output.png')

    model_outputs = img_to_3d(runner.run_segmentation, runner.run_zero123plus1_2, runner.run_zero123plus1_2_to_mesh, image=image)

if __name__ == '__main__':
    main()