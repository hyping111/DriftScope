import sys
sys.path.append("..")  # to import dexter from notebooks/ directory
import torch
from diffusers import DiffusionPipeline
from dexter.pipelines import TextPipeline

pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.bfloat16,
    variant=None,
    safety_checker=None,
    requires_safety_checker=False,
)
#"sd2-community/stable-diffusion-2-1",
#"CompVis/stable-diffusion-v1-4"
pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.bfloat16,
    variant=None,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe = DiffusionPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-1",
    torch_dtype=torch.bfloat16,
    variant=None,
    safety_checker=None,
    requires_safety_checker=False,
)

text_encoder = TextPipeline(
    translation_matrix=None,
    n_soft_prompt=1,
    clip_checkpoint="openai/clip-vit-large-patch14",
    clip_subfolder_encoder="",  # yes, it's empty
    clip_subfolder_tokenizer="",
)
# sd1.5 shares same text encoder as sd1.4, so we can reuse it
text_encoder = TextPipeline(
    translation_matrix=None,
    n_soft_prompt=1,
    clip_checkpoint="sd2-community/stable-diffusion-2-1",
    clip_subfolder_encoder="text_encoder",
    clip_subfolder_tokenizer="tokenizer",
)

text_encoder = TextPipeline(
    translation_matrix=None,
    n_soft_prompt=1,
    clip_checkpoint="openai/clip-vit-large-patch14",
    clip_subfolder_encoder="",  # yes, it's empty
    clip_subfolder_tokenizer="",
)
# sd1.5 shares same text encoder as sd1.4, so we can reuse it
text_encoder = TextPipeline(
    translation_matrix=None,
    n_soft_prompt=1,
    clip_checkpoint="sd2-community/stable-diffusion-2-1",
    clip_subfolder_encoder="text_encoder",
    clip_subfolder_tokenizer="tokenizer",
)