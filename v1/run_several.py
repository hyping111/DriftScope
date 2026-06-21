# import sys
# sys.path.append("..")
from dexter import DEXTER, RunConfig, build_classifier_and_transforms
import torch
import argparse


parser = argparse.ArgumentParser(description='DEXTER model configuration')
parser.add_argument(
    '--method',
    type=str,
    default='db15',
    help='Method to use'
)
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
parser.add_argument(
    '--gradient_checkpointing',
    action='store_true',
    help='Whether to enable gradient checkpointing for the UNet (saves VRAM at the cost of more computation)',
)
args = parser.parse_args()


device = "cuda"
# cfg = RunConfig(class_to_activate=291, classifier="robust50", optim_steps=100)
cfg = RunConfig(optim_steps=100)
#"sd2-community/stable-diffusion-2-1",
#"CompVis/stable-diffusion-v1-4"
cfg.torch_dtype = 'bfloat16'
cfg.variant = None
cfg.outdir = f'./outputs/{args.method}{"/" + args.out_subfolder if args.out_subfolder else ""}'

cfg.objective_direction = 'maximize'
cfg.mask_type = 'single_mask'
# cfg.mask_type = 'multi_mask'
cfg.num_masks = 6 # when using multi_mask

# cfg.bs = 1
cfg.bs = 4
cfg.diff_steps = 50
cfg.guidance_scale = 7.5
cfg.gen_dim = 512
cfg.gen_steps = args.gen_steps
cfg.seed = 42
cfg.seed = list(range(42, 4200, 4))[13]
# cfg.seed = None
cfg.save_images =False
cfg.convergence_threshold = -4  # grows with gen_steps
cfg.prompt = args.prompt
cfg.gradient_checkpointing = args.gradient_checkpointing

# 3 steps for full img
# cfg.outdir = f'out/{args.method}/3-gen-steps-full'
# cfg.diff_steps = 3
# cfg.gen_steps = None


if args.method == 'ac':
    # Ablating Concepts - unlearning
    cfg.base_model = "CompVis/stable-diffusion-v1-4"
    
    cfg.modified_model = "/DriftScope/diffusion_modified_models/unlearning/sd14/nudity/AC/AC-Nudity-Diffusers-UNet-xattn.pt"
    cfg.load_type = 'weights'
    cfg.modified_model_strict_load = False

    cfg.n_heads=8
    cfg.input_dim_att_layer=64
    cfg.att_blocks =["down_blocks.0","up_blocks.3"]
    cfg.N_layers=5
    

elif args.method == 'spm':
    # SPM - unlearning
    cfg.base_model = "CompVis/stable-diffusion-v1-4"
    cfg.modified_model = "/DriftScope/diffusion_modified_models/unlearning/sd14/nudity/SPM/SPM-Nudity-Diffusers-UNet.pt"
    cfg.load_type = 'weights'
    cfg.modified_model_strict_load = True

    cfg.n_heads=8
    cfg.input_dim_att_layer=64
    cfg.att_blocks =["down_blocks.0","up_blocks.3"]
    cfg.N_layers=5
    

elif args.method == 'sh':
    # SPM - unlearning
    cfg.base_model = "CompVis/stable-diffusion-v1-4"
    cfg.modified_model = "/DriftScope/diffusion_modified_models/unlearning/sd14/nudity/Scissorhands/Scissorhands-Nudity-Diffusers-UNet.pt"
    cfg.load_type = 'weights'
    cfg.modified_model_strict_load = True

    cfg.n_heads=8
    cfg.input_dim_att_layer=64
    cfg.att_blocks =["down_blocks.0","up_blocks.3"]
    cfg.N_layers=5
    
elif args.method == 'db21':
    # DreamBooth - customization
    cfg.base_model = "sd2-community/stable-diffusion-2-1"
    cfg.modified_model = "/DriftScope/diffusion_modified_models/customization/sd21/customconcept101/dblora-prior/furniture_sofa1/pytorch_lora_weights.safetensors"
    cfg.load_type = 'lora'

    cfg.base_model = "sd2-community/stable-diffusion-2-1"
    cfg.clip_orig_text_encoder_name = "sd2-community/stable-diffusion-2-1"
    cfg.clip_orig_text_encoder_name_subfolder = "text_encoder"
    cfg.clip_orig_tokenizer_name_subfolder = "tokenizer"
    cfg.load_type = 'lora'

    #if args.comb == 1 :
    #    cfg.n_heads=5
    #    cfg.input_dim_att_layer=64
    #    cfg.att_blocks =["down_blocks.0","up_blocks.3"]
    #    cfg.N_layers=5

    cfg.n_heads=20#5
    cfg.input_dim_att_layer=16
    cfg.att_blocks =["down_blocks.2","up_blocks.1"]
    cfg.N_layers=5

elif args.method == 'db15':
    # DreamBooth - customization
    cfg.base_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    cfg.modified_model = "/DriftScope/diffusion_modified_models/customization/sd15/customconcept101/dbft-prior/pet_dog4/unet/diffusion_pytorch_model.safetensors"
    cfg.load_type = 'weights'
    cfg.modified_model_strict_load = True

    cfg.n_heads=8
    cfg.input_dim_att_layer=64
    cfg.att_blocks =["down_blocks.0","up_blocks.3"]
    cfg.N_layers=5

    
elif args.method == 'cd':
    # Custom Diffusion with LoRA - customization
    cfg.base_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    cfg.modified_model = "/DriftScope/diffusion_modified_models/customization/sd15/customconcept101/cdlora-prior/pet_dog4"
    cfg.load_type = 'lora'
else:
    raise ValueError(f"Unknown method: {args.method}")


clf, tfms = None, None

dexter = DEXTER(cfg, device=device)

run_dir = dexter.find_unique_masks(num_masks_to_find=100)


