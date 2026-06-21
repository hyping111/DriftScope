# import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput#, VaeImageProcessor
from diffusers.utils import (
    # USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    # replace_example_docstring,
    # scale_lora_layers,
    # unscale_lora_layers,
)

from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput


from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import EXAMPLE_DOC_STRING, rescale_noise_cfg, retrieve_timesteps

import torch.nn.functional as F
from torch import nn

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

#new
from diffusers.models.attention_processor import AttnProcessor
from einops import rearrange, reduce,repeat
class IPAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        if 'kwargs' in kwargs and 'attn_map_buffer' in kwargs['kwargs']:
            enable_grad = kwargs['kwargs']['enable_grad'] if 'enable_grad' in kwargs['kwargs'] else False
            with torch.set_grad_enabled(enable_grad):
                kwargs['kwargs']['attn_map_buffer'].append(attn.get_attention_scores(query.reshape(-1, query.shape[2], query.shape[3]), key.reshape(-1, key.shape[2], key.shape[3]), attention_mask))  # [B*heads, H*W, token_len]

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        activate_ip_adapter = False#True
        if 'kwargs' in kwargs and 'iteration_activation_steps' in kwargs['kwargs']:
            activate_ip_adapter = kwargs['kwargs']['iteration_counter'][self.name] in kwargs['kwargs']['iteration_activation_steps']
        if activate_ip_adapter:
            # for ip-adapter
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if 'key_buffer' in kwargs['kwargs']:
                kwargs['kwargs']['key_buffer'].append(key)
            if 'ip_key_buffer' in kwargs['kwargs']:
                kwargs['kwargs']['ip_key_buffer'].append(ip_key)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            ip_attn = kwargs['kwargs']['ip_attn'] if 'kwargs' in kwargs and 'ip_attn' in kwargs['kwargs'] else 'token-0'
            if ip_attn == 'all_tokens':
                ip_hidden_states = F.scaled_dot_product_attention(
                    query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
            elif ip_attn.startswith('token-'):
                token_idx = int(ip_attn.split('-')[1])
                ip_hidden_states = F.scaled_dot_product_attention(
                    query, ip_key[:, :, token_idx, :].unsqueeze(2), ip_value[:, :, token_idx, :].unsqueeze(2), attn_mask=None, dropout_p=0.0, is_causal=False
                )
            else:
                raise ValueError('Invalid ip_attn value')

            if 'kwargs' in kwargs and 'ip_attn_map_buffer' in kwargs['kwargs']:
                enable_grad = kwargs['kwargs']['enable_grad'] if 'enable_grad' in kwargs['kwargs'] else False
                with torch.set_grad_enabled(enable_grad):
                    kwargs['kwargs']['ip_attn_map_buffer'].append(attn.get_attention_scores(query.reshape(-1, query.shape[2], query.shape[3]), ip_key.reshape(-1, ip_key.shape[2], ip_key.shape[3]), attention_mask=None))  # [B*heads, H*W, token_len]

            # print('ip_hidden_states bef', ip_hidden_states.shape)  [2, 20, 1024, 64]
            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states = ip_hidden_states.to(query.dtype)
            # print('ip_hidden_states after', ip_hidden_states.shape)  [2, 1024, 1280]

            orig_attn_maps = attn.get_attention_scores(query.reshape(-1, query.shape[2], query.shape[3]), key.reshape(-1, key.shape[2], key.shape[3]), attention_mask)
            # print('orig_attn_maps', orig_attn_maps.shape)  # [40, 1024, 77]
            if batch_size == 2:  # bs=1 x cond=2
                orig_attn_maps = rearrange(orig_attn_maps, '(cond heads) img_len token_len -> 1 cond heads img_len token_len', cond=2)
            else:
                orig_attn_maps = rearrange(orig_attn_maps, '(cond b heads) img_len token_len -> b cond heads img_len token_len', cond=2, b=batch_size // 2)
            obj_attn_map = orig_attn_maps[:, :, :, :, kwargs['kwargs']['selected_prompt_token_idx']].unsqueeze(-1)
            # print('obj_attn_map', obj_attn_map.shape)  # [b, 2, 20, 1024, 1]
            obj_attn_map = obj_attn_map[:, 1, :, :, :].unsqueeze(1)  # keep the the conditional part of the guidance [b, 1, 20, 1024, 1]. TODO do CFG maybe.
            # print('obj_attn_map af selec cond', obj_attn_map.shape)
            obj_attn_map = obj_attn_map.mean(dim=2)  # [b, 1, 1024, 1]
            # print('obj_attn_map af mean', obj_attn_map.shape)
            if 'kwargs' in kwargs and 'cutoff_perc' in kwargs['kwargs']:
                # set to 0 the N-th lowest percentage values over the spatial dimensions
                perc = kwargs['kwargs']['cutoff_perc']  # 0.8
                # obj_attn_map = obj_attn_map * (obj_attn_map >= torch.quantile(obj_attn_map.float(), perc, dim=2, keepdim=True))
                # obj_attn_map = obj_attn_map.to(ip_hidden_states.dtype)
                obj_attn_map = obj_attn_map >= torch.quantile(obj_attn_map.float(), perc, dim=2, keepdim=True)
            # fill the cond since we discarded the unconditional one
            # print('obj_attn_map bef repeat', obj_attn_map.shape)
            obj_attn_map = repeat(obj_attn_map, 'b cond img_len token_len -> b (cond repeat) img_len token_len', repeat=2)
            # print('obj_attn_map af repeat', obj_attn_map.shape)
            # fold the batch size back into the condition
            obj_attn_map = rearrange(obj_attn_map, 'b cond img_len token_len -> (cond b) img_len token_len')
            # print('obj_attn_map af fold', obj_attn_map.shape)
            ip_mask = obj_attn_map

            if 'kwargs' in kwargs and 'ip_mask_buffer' in kwargs['kwargs']:
                kwargs['kwargs']['ip_mask_buffer'].append(ip_mask)

            hidden_states = hidden_states + self.scale * ip_hidden_states * ip_mask

        #     print(self.name, kwargs['kwargs']['iteration_counter'][self.name], 'yes')
        # else:
        #     print(self.name, kwargs['kwargs']['iteration_counter'][self.name], 'no')

        if 'kwargs' in kwargs and 'iteration_counter' in kwargs['kwargs']:
            kwargs['kwargs']['iteration_counter'][self.name] += 1

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
#
def generate_two_branches(
        pipe_base,
        pipe_modified,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        cfg,
        dim: int = 512,
        guidance_scale: float = 1.0,
        seed: int = None,
    ):
        """Generate a batch of latents and decoded images given conditional text embeddings.

        Args:
            cond: Conditional text embeddings [1, seq, hidden].
            uncond: Unconditional embeddings [1, seq, hidden].
            cfg: Run configuration including batch size and diffusion steps.
            dim: Spatial size of output image (latent is dim/8).
            guidance_scale: CFG scale for classifier-free guidance.

        Returns:
            Tuple (latents, img_tensor) where img_tensor is decoded to [B,3,H,W].
        """
        assert cond.shape[0] == 1, f"TODO: If you use more than one prompt, revise cfg.bs and num_images_per_prompt param settings for correct generation. ({cond.shape[0]=} {cfg.bs=})"

        #new
        attn_map_buffer_base = []
        ip_attn_map_buffer_base = []
        attn_map_buffer_mod = []
        ip_attn_map_buffer_mod = []
        N_layers=cfg.N_layers
        input_dim=cfg.input_dim_att_layer
        n_heads=cfg.n_heads 
        diff_steps = cfg.gen_steps


        with torch.no_grad():
            latents_base = run_pipe_call(
                pipe_base,
                prompt_embeds=cond,
                negative_prompt_embeds=uncond,
                num_inference_steps=cfg.diff_steps,
                height=dim,
                width=dim,
                guidance_scale=guidance_scale,
                output_type="latent",
                generator=torch.Generator().manual_seed(seed) if seed is not None else None,
                gen_cut_step=cfg.gen_steps,
                num_images_per_prompt=cfg.bs,
                cross_attention_kwargs={'kwargs': {'attn_map_buffer': attn_map_buffer_base, 'ip_attn_map_buffer': ip_attn_map_buffer_base}},
                att_blocks=cfg.att_blocks,
            )[0]

        print(torch.stack(attn_map_buffer_base).shape)
        kk_base = rearrange(torch.stack(attn_map_buffer_base), '(denoising_steps layers) (b cond heads) (h w) token_len  -> b denoising_steps layers cond heads token_len h w', denoising_steps=diff_steps, b=4, layers=N_layers, cond=2, heads=n_heads, h=input_dim, w=input_dim)
        print(kk_base.shape)
        kk_base = reduce(kk_base, 'b denoising_steps layers cond heads token_len h w -> b denoising_steps layers cond token_len h w', 'mean')
        kk_base = reduce(kk_base, 'b denoising_steps layers cond token_len h w -> b denoising_steps cond token_len h w', 'mean')
        kk_base= kk_base[:,:,1]


        attn_map_buffer_base = []

        latents_modified = run_pipe_call(
            pipe_modified,
            prompt_embeds=cond,
            negative_prompt_embeds=uncond,
            num_inference_steps=cfg.diff_steps,
            height=dim,
            width=dim,
            guidance_scale=guidance_scale,
            output_type="latent",
            generator=torch.Generator().manual_seed(seed) if seed is not None else None,
            gen_cut_step=cfg.gen_steps,
            num_images_per_prompt=cfg.bs,
            cross_attention_kwargs={'kwargs': {'attn_map_buffer': attn_map_buffer_mod, 'ip_attn_map_buffer': ip_attn_map_buffer_mod}},
            att_blocks=cfg.att_blocks,
        )[0]

        kk_mod = rearrange(torch.stack(attn_map_buffer_mod), '(denoising_steps layers) (b cond heads) (h w) token_len  -> b denoising_steps layers cond heads token_len h w', denoising_steps=diff_steps, b=4, layers=N_layers, cond=2, heads=n_heads, h=input_dim, w=input_dim)
        kk_mod = reduce(kk_mod, 'b denoising_steps layers cond heads token_len h w -> b denoising_steps layers cond token_len h w', 'mean')
        kk_mod = reduce(kk_mod, 'b denoising_steps layers cond token_len h w -> b denoising_steps cond token_len h w', 'mean')
        kk_mod= kk_mod[:,:,1]
    

        attn_map_buffer_mod = []
        
        return latents_base, latents_modified,kk_base,kk_mod


def run_pipe_call(
    self,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    sigmas: List[float] = None,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    att_blocks: List[str]= None,
    guidance_rescale: float = 0.0,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    gen_cut_step = None,
    **kwargs,
):
    r"""
    The call function to the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
        height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        sigmas (`List[float]`, *optional*):
            Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
            their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
            will be used.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            A higher guidance scale value encourages the model to generate images closely linked to the text
            `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide what to not include in image generation. If not defined, you need to
            pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) from the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only
            applies to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        latents (`torch.Tensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
            provided, text embeddings are generated from the `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
            not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
        ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
        ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
            Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
            IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
            contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
            provided, embeddings are computed from the `ip_adapter_image` input argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generated image. Choose between `PIL.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
            [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when
            using zero terminal SNR.
        clip_skip (`int`, *optional*):
            Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
            the output of the pre-final layer will be used for computing the prompt embeddings.
        callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
            A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
            each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
            DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
            list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
            otherwise a `tuple` is returned where the first element is a list with the generated images and the
            second element is a list of `bool`s indicating whether the corresponding generated image contains
            "not-safe-for-work" (nsfw) content.
    """
    #new
    def set_ip_adapter(self,att_blocks):
        unet = self.unet
        attn_procs = {}
        
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]


            target_blocks = att_blocks
            if cross_attention_dim is None or not any([t in name for t in target_blocks]):
                attn_procs[name] = AttnProcessor()
          
            else:
                num_tokens=0
                attn_procs[name] = IPAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=num_tokens,
                    # name=name,
                ).to(self.device, dtype=torch.float16)
                attn_procs[name].name = name
        unet.set_attn_processor(attn_procs)
       

    
    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    #new 0.0
    set_ip_adapter(self,att_blocks)
    #
    # 0. Default height and width to unet

    if not height or not width:
        height = (
            self.unet.config.sample_size
            if self._is_unet_config_sample_size_int
            else self.unet.config.sample_size[0]
        )
        width = (
            self.unet.config.sample_size
            if self._is_unet_config_sample_size_int
            else self.unet.config.sample_size[1]
        )
        height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
    # to deal with lora scaling and other possible forward hooks

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        ip_adapter_image,
        ip_adapter_image_embeds,
        callback_on_step_end_tensor_inputs,
    )

    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # 3. Encode input prompt
    lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
    )

    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        self.do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=self.clip_skip,
    )

    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            self.do_classifier_free_guidance,
        )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, timesteps, sigmas
    )

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 6.1 Add image embeds for IP-Adapter
    added_cond_kwargs = (
        {"image_embeds": image_embeds}
        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
        else None
    )

    # 6.2 Optionally get Guidance Scale Embedding
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            if hasattr(self.scheduler, "scale_model_input"):
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://huggingface.co/papers/2305.08891
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

            if XLA_AVAILABLE:
                xm.mark_step()
            
            if gen_cut_step is not None and (i + 1) >= gen_cut_step:
                break

    if not output_type == "latent":
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
            0
        ]
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
