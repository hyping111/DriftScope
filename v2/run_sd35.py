import os
import sys
from dexter.dexter_sd3 import DEXTER, RunConfig
import torch
import argparse
import torch.nn as nn
parser = argparse.ArgumentParser(description='DEXTER model configuration')

parser.add_argument(
    '--prompt',
    type=str,
    default='A picture of a',
)
parser.add_argument(
    '--out_subfolder',
    type=str,
    default=None,
)
parser.add_argument(
    '--gen_steps',
    type=int,
    default=4,
)

args = parser.parse_args()



device = "cuda"
# cfg = RunConfig(class_to_activate=291, classifier="robust50", optim_steps=100)
cfg = RunConfig(optim_steps=100)
cfg.prompt = args.prompt
cfg.outdir = f'./outputs/dexter_outs/time{"/" + args.out_subfolder if args.out_subfolder else ""}'

# sdxl
cfg.base_model = "stabilityai/stable-diffusion-3.5-medium"
cfg.clip_orig_text_encoder_name = cfg.base_model
cfg.clip_orig_text_encoder_name_subfolder = "text_encoder"
cfg.clip_orig_tokenizer_name_subfolder = "tokenizer"
cfg.clip_orig_text_encoder_name_2 = cfg.base_model
cfg.clip_orig_text_encoder_name_subfolder_2 = "text_encoder_2"
cfg.clip_orig_tokenizer_name_subfolder_2 = "tokenizer_2"

cfg.torch_dtype = 'bf16'
cfg.torch_dtype = 'fp16'
cfg.variant = None

cfg.objective_direction = 'maximize'
cfg.mask_type = 'single_mask'
# cfg.mask_type = 'multi_mask'
cfg.num_masks = 6 # when using multi_mask

cfg.bs = 1
#cfg.bs = 4
cfg.diff_steps = 28#
cfg.guidance_scale = 7.0
cfg.gen_dim = 1024
# cfg.gen_dim = 512  # due to OOM
cfg.gen_steps = 4
cfg.seed = 42
cfg.seed = list(range(42, 4200, 4))[13]
cfg.save_images = False
cfg.convergence_threshold = -4  # grows with gen_steps

cfg.gradient_checkpointing = True


# sd35m

# dblora
cfg.modified_model = "/DriftScope/diffusion_modified_models/customization/sd35m/customconcept101/dblora-prior/pet_dog4"
cfg.load_type = 'lora'


# clf, tfms = build_classifier_and_transforms(cfg.classifier, device=device, use_tfms=True, weights="../checkpoints/robust_resnet50.pth")
# clf, tfms = torch.nn.L1Loss(), None
clf, tfms = None, None

cfg.n_heads=24
cfg.input_dim_att_layer=64
cfg.att_blocks =["transformer_blocks.0.attn.processor", "transformer_blocks.23.attn.processor"]
cfg.N_layers=2




dexter = DEXTER(cfg, device=device)
run_dir = dexter.find_unique_masks(num_masks_to_find=100)
#run_dir = dexter.analyze_classifier(clf, tfms)