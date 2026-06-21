"""Core DEXTER orchestration logic for prompt optimization and analysis."""

import logging
import os
from statistics import mean
from typing import Optional

import torch
from PIL import Image
from sentence_transformers.util import dot_score, normalize_embeddings, semantic_search
from tqdm.auto import tqdm

from .pipelines import VisualPipeline, TextPipeline, TextPipelineWithProjection
from .pipelines.sd3_modified_att import generate_two_branches
from .config import DISALLOWED_TARGET_CHARS, RunConfig
from .utils import (
    change_target,
    ensure_output_dir,
    init_diffusion_pipeline,
    initialize_dynamic_target,
    latents_to_pil,
    latents_to_torch_img,
    load_translation_matrix,
    select_top_features,
    set_random_seed,
    text_enc,
    text_enc_with_proj,
    forward_with_features,
    update_target,
    validate_options,
)
from diffusers import DiffusionPipeline
from safetensors.torch import load_file


# Silence noisy framework logging in the CLI.
logging.disable(logging.WARNING)


def load_weights_file(path: str, device: str = 'cpu', weights_only: bool = True) -> dict:
    """Load weights from .pt or .safetensors file."""
    if path.endswith('.safetensors'):
        return load_file(path)
    return torch.load(path, map_location=device, weights_only=weights_only)

class DEXTER:
    """Orchestrates the prompt optimization and classifier analysis workflow."""

    def __init__(self, config: RunConfig, device: str = "cuda") -> None:
        """Set up diffusion/text pipelines, output folders, and bookkeeping."""
        self.config = config
        self.device = device
        self.feat_to_activate = select_top_features(
            config.classifier, config.class_to_activate
        )

        # set_random_seed(config.seed)

        # self.pipe, self.tokenizer, self.clip_orig_text_encoder = (
        #     init_diffusion_pipeline(device)
        # )

        torch_dtype = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
        }.get(config.torch_dtype, torch.float16)

        self.pipe_base = DiffusionPipeline.from_pretrained(
            config.base_model,
            text_encoder_3=None,
            tokenizer_3=None,
            # quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            variant=config.variant,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=False,
        ).to(self.device)
        self.tokenizer = self.pipe_base.tokenizer
        self.tokenizer_2 = self.pipe_base.tokenizer_2
        from transformers import CLIPTextModelWithProjection
        self.clip_orig_text_encoder_name = getattr(config, "clip_orig_text_encoder_name", "openai/clip-vit-large-patch14")
        self.clip_orig_text_encoder_name_subfolder = getattr(config, "clip_orig_text_encoder_name_subfolder", "")
        self.clip_orig_tokenizer_name_subfolder = getattr(config, "clip_orig_tokenizer_name_subfolder", "")
        self.clip_orig_text_encoder = (
            CLIPTextModelWithProjection.from_pretrained(self.clip_orig_text_encoder_name, subfolder=self.clip_orig_text_encoder_name_subfolder).to(self.device).half()
        )
        self.clip_orig_text_encoder_name_2 = getattr(config, "clip_orig_text_encoder_name_2", "openai/clip-vit-large-patch14")
        self.clip_orig_text_encoder_name_subfolder_2 = getattr(config, "clip_orig_text_encoder_name_subfolder_2", "")
        self.clip_orig_tokenizer_name_subfolder_2 = getattr(config, "clip_orig_tokenizer_name_subfolder_2", "")
        self.clip_orig_text_encoder_2 = (
            CLIPTextModelWithProjection.from_pretrained(self.clip_orig_text_encoder_name_2, subfolder=self.clip_orig_text_encoder_name_subfolder_2).to(self.device).half()
        )

        self.pipe_modified = DiffusionPipeline.from_pretrained(
            config.base_model,
            text_encoder_3=None,
            tokenizer_3=None,
            # quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            variant=config.variant,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        ).to(self.device)
        if config.load_type == 'weights':
            weights = load_weights_file(config.modified_model)
            self.pipe_modified.transformer.load_state_dict(weights, strict=config.modified_model_strict_load)
            print(f"Loaded transformer weights from {config.modified_model}")
        elif config.load_type == 'lora':
            self.pipe_modified.load_lora_weights(config.modified_model)
            print(f"Loaded LoRA weights from {config.modified_model}")
        else:
            raise ValueError(f"Unknown load_type: {config.load_type}")

        try:
            self.translation_matrix = load_translation_matrix(
                clip_checkpoint=self.clip_orig_text_encoder_name,
                bert_checkpoint="google-bert/bert-base-uncased",
                clip_subfolder_encoder=self.clip_orig_text_encoder_name_subfolder,
                clip_subfolder_tokenizer=self.clip_orig_tokenizer_name_subfolder,
            )
        except Exception as e:
            print(f"Warning: Translation Matrix not found, it will be created")
            self.translation_matrix = None
        
        try:
            self.translation_matrix_2 = load_translation_matrix(
                clip_checkpoint=self.clip_orig_text_encoder_name_2,
                bert_checkpoint="google-bert/bert-base-uncased",
                clip_subfolder_encoder=self.clip_orig_text_encoder_name_subfolder_2,
                clip_subfolder_tokenizer=self.clip_orig_tokenizer_name_subfolder_2,
            )
        except Exception as e:
            print(f"Warning: Translation Matrix not found, it will be created")
            self.translation_matrix_2 = None

        self.vae = self.pipe_base.vae
        # self.scheduler = self.pipe.scheduler
        # self.unet = self.pipe.unet
        # self.visual_pipeline = VisualPipeline(
        #     self.scheduler, self.unet, self.vae, device=device
        # )

        self.out_folder = ensure_output_dir(
            config.outdir, config.class_to_activate, config.to_kwargs()
        )
        validate_options(config.sp_init, config.no_match_uniform, config.mask_type)

        self.frame_folder = os.path.join(self.out_folder, "frames")
        os.makedirs(self.frame_folder, exist_ok=True)

        self.previous_target_words = []



        if getattr(config, "gradient_checkpointing", False):
            self.pipe_base.transformer.enable_gradient_checkpointing()
            self.pipe_modified.transformer.enable_gradient_checkpointing()

    @staticmethod
    def _format_words(words) -> str:
        """Readable string for optional/iterable word collections."""
        if words in (None, ""):
            return "-"
        if isinstance(words, (list, tuple, set)):
            if len(words) == 0:
                return "-"
            return ", ".join(str(w) for w in words)
        return str(words)

    @staticmethod
    def _print_section(title: str) -> None:
        """Pretty-print a section header for CLI logging."""
        line = "=" * 60
        print(f"\n{line}\n{title}\n{line}")

    @staticmethod
    def _print_subsection(title: str) -> None:
        """Pretty-print a subsection header for CLI logging."""
        line = "-" * 50
        print(f"\n{line}\n{title}\n{line}")

    @staticmethod
    def _print_line(label: str, value) -> None:
        """Consistent, padded key/value output."""
        print(f"{label:<18} {value}")

    def analyze_classifier(
        self, target_classifier, transforms: Optional[torch.nn.Module] = None
    ) -> str:
        """Optimize text prompts against the classifier and persist artifacts.

        Args:
            target_classifier: Torch vision model used for scoring generated images.
            transforms: Optional preprocessing applied before classifier forward.

        Returns:
            Path to the run output folder containing frames, prompts, and logs.
        """
        cfg = self.config
        feat_to_activate = self.feat_to_activate
        device = self.device

        dummy_prompt = "X " * cfg.n_tokens
        dummy_prompt = dummy_prompt.strip()

        if cfg.prefix not in ["", None]:
            prompts = [cfg.prefix + " " + dummy_prompt]
        else:
            prompts = [dummy_prompt]

        dim = cfg.gen_dim
        g = cfg.guidance_scale #1.0
        diff_seed = cfg.seed

        captions = []
        convergence = 0

        for outer_idx in range(cfg.topk):

            self.text_encoder = TextPipelineWithProjection(translation_matrix=self.translation_matrix, n_soft_prompt=self.config.n_soft_prompt, clip_checkpoint=self.clip_orig_text_encoder_name, clip_subfolder_encoder=self.clip_orig_text_encoder_name_subfolder, clip_subfolder_tokenizer=self.clip_orig_tokenizer_name_subfolder)
            self.text_encoder = self.text_encoder.to(device).half()
            self.text_encoder_2 = TextPipelineWithProjection(translation_matrix=self.translation_matrix_2, n_soft_prompt=self.config.n_soft_prompt, clip_checkpoint=self.clip_orig_text_encoder_name_2, clip_subfolder_encoder=self.clip_orig_text_encoder_name_subfolder_2, clip_subfolder_tokenizer=self.clip_orig_tokenizer_name_subfolder_2)
            self.text_encoder_2 = self.text_encoder_2.to(device).half()

            self.text_encoder_2.bert = self.text_encoder.bert  # same BERT and same soft prompt params

            self.optimizer = torch.optim.AdamW(self.text_encoder.prompt_parameters, lr=self.config.lr, eps=1e-4 )

            for inner_optim_step in range(cfg.optim_steps):
                self._print_subsection(
                    f"Step {inner_optim_step + 1}/{cfg.optim_steps}"
                )
                if len(captions) == 0:
                    prompt = "A picture of a"

                if inner_optim_step == 0:
                    target_words = initialize_dynamic_target(cfg.mask_type)

                label = update_target(target_words, prompt, cfg.mask_type)

                self._print_line("Target prompt:", label)
                text, _ = text_enc_with_proj(prompts, self.tokenizer, self.clip_orig_text_encoder, device=device)
                text_2, _ = text_enc_with_proj(prompts, self.tokenizer_2, self.clip_orig_text_encoder_2, device=device)
                # pred_words = [""] * 6

                cond_1, pooled_cond_1, masked_loss_1, pred_word = self.text_encoder(
                    prompt=prompt,
                    label=label,
                    tau=cfg.tau,
                    mask_type=cfg.mask_type,
                    target_words=target_words,
                )
                cond_2, pooled_cond_2, masked_loss_2, pred_word_2 = self.text_encoder_2(
                    prompt=prompt,
                    label=label,
                    tau=cfg.tau,
                    mask_type=cfg.mask_type,
                    target_words=target_words,
                )
                clip_prompt_embeds = torch.cat([cond_1, cond_2], dim=-1)  # concat CLIP embeddings from both encoders
                t5_prompt_embed = torch.zeros(  # recreate no T5 use
                    (
                        1, # batch_size * num_images_per_prompt,
                        self.pipe_base.tokenizer_max_length,
                        self.pipe_base.transformer.config.joint_attention_dim,
                    ),
                    device=device,
                    dtype=self.pipe_base.dtype,
                )
                clip_prompt_embeds = torch.nn.functional.pad(  # Pad CLIP prompt embeds to match T5 dimension as in the paper
                    clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
                )
                cond = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)  # Final conditioning combines CLIP and T5 (though T5 is not used here)
                pooled_cond = torch.cat([pooled_cond_1, pooled_cond_2], dim=-1)  # Final pooled conditioning also concatenates both encoders
                masked_loss = (masked_loss_1 + masked_loss_2) / 2

                pred_words = pred_word
                self._print_line("Predicted words:", self._format_words(pred_word))
                uncond_1, pooled_uncond_1 = text_enc_with_proj(
                    [""],
                    self.tokenizer,
                    self.clip_orig_text_encoder,
                    text.shape[1],
                    device=device,
                )
                uncond_2, pooled_uncond_2 = text_enc_with_proj(
                    [""],
                    self.tokenizer_2,
                    self.clip_orig_text_encoder_2,
                    text_2.shape[1],
                    device=device,
                )
                negative_clip_prompt_embeds = torch.cat([uncond_1, uncond_2], dim=-1)
                t5_negative_prompt_embed = torch.zeros(  # recreate no T5 use
                    (
                        1, # batch_size * num_images_per_prompt,
                        self.pipe_base.tokenizer_max_length,
                        self.pipe_base.transformer.config.joint_attention_dim,
                    ),
                    device=device,
                    dtype=self.pipe_base.dtype,
                )
                negative_clip_prompt_embeds = torch.nn.functional.pad(  # Pad CLIP prompt embeds to match T5 dimension as in the paper
                    negative_clip_prompt_embeds, (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1])
                )
                uncond = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
                pooled_uncond = torch.cat([pooled_uncond_1, pooled_uncond_2], dim=-1)  # Final pooled conditioning also concatenates both encoders

                # generate images
                # latents, img = self.visual_pipeline.generate(
                #     cond, uncond, cfg, dim=dim, guidance_scale=g
                # )
                latents_base, latents_modified,kk_base,kk_mod  = generate_two_branches(
                    self.pipe_base, self.pipe_modified, cond, uncond, pooled_cond, pooled_uncond, cfg, dim=dim, guidance_scale=g, seed=diff_seed,
                )
                # classifier forward
                # if cfg.use_tfms and transforms is not None:
                #     img = transforms(img)
                # output, features = forward_with_features(target_classifier, img)
                # if cfg.mask_type == "multi_mask" and features is None:
                #     raise RuntimeError(
                #         "Multi-mask mode requires feature tensors, "
                #         "but the classifier did not return any."
                #     )

                # loss computation
                # ce_loss = torch.nn.CrossEntropyLoss()(
                #     output,
                #     torch.tensor([cfg.class_to_activate] * img.shape[0]).to(device),
                # )
                # predicted_class = torch.argmax(output, dim=1).cpu().numpy()

                # if predicted_class[0] == cfg.class_to_activate:
                #     pil_image = latents_to_pil(latents.detach(), self.vae)[0]
                #     pil_image.save(os.path.join(self.frame_folder, f"{inner_optim_step}.png"))
                ce_loss = torch.nn.functional.l1_loss(kk_mod,kk_base.detach())
                #ce_loss = torch.nn.functional.l1_loss(latents_modified, latents_base.detach())
                if cfg.objective_direction == 'maximize':
                    ce_loss = -ce_loss
                features = None
                predicted_class = [-1]  # Dummy value since we don't have classifier output
                # latent_loss = torch.nn.functional.mse_loss(latents_modified, latents_base.detach())

                # max_losses = []
                # if cfg.mask_type == "multi_mask":
                #     if cfg.classifier == "vit_b_16":
                #         feature_slices = [features[:, :, idx] for idx in feat_to_activate]
                #     else:
                #         feature_slices = [features[:, idx] for idx in feat_to_activate]

                #     max_losses = [-feat_slice.mean() for feat_slice in feature_slices]

                if inner_optim_step == 0:
                    if cfg.mask_type == "multi_mask":
                        # best_loss = [[ce_loss.item()]] + [[loss.item()] for loss in max_losses]
                        best_loss = [[ce_loss.item()]]
                    else:
                        best_loss = [ce_loss.item()]

                if cfg.mask_type == "multi_mask":
                    self._print_line("Masked loss:", f"{masked_loss.item():.4f}")
                    self._print_line("Class loss:", f"{ce_loss.item():.4f}")
                    # feature_losses = " | ".join(
                    #     f"F{idx}={m_loss.item():.4f}"
                    #     for idx, m_loss in enumerate(max_losses, 1)
                    # )
                    best_losses = " | ".join(
                        f"B{idx}={torch.mean(torch.tensor(b_loss)):.4f}"
                        for idx, b_loss in enumerate(best_loss, 1)
                    )
                    # self._print_line("Feature losses:", feature_losses)
                    self._print_line("Best losses:", best_losses)
                else:
                    self._print_line("Masked loss:", f"{masked_loss.item():.4f}")
                    self._print_line("Class loss:", f"{ce_loss.item():.4f}")
                    self._print_line(
                        "Best loss:", f"{torch.mean(torch.tensor(best_loss)):.4f}"
                    )

                # if cfg.mask_type == "multi_mask":
                #     loss = ce_loss + sum(max_losses) + masked_loss
                # else:
                #     loss = ce_loss + masked_loss
                loss = ce_loss + masked_loss

                self._print_line(
                    "Step summary:",
                    f"loss={loss.item():.4f} | pred class={predicted_class}",
                )

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if cfg.mask_type == "multi_mask":
                    # target_words = list(target_words)

                    # (
                    #     ce_history,
                    #     f1_history,
                    #     f2_history,
                    #     f3_history,
                    #     f4_history,
                    #     f5_history,
                    # ) = best_loss

                    # ce_loss_improved = ce_loss.item() < torch.mean(
                    #     torch.tensor(ce_history)
                    # )
                    # loss_improvements = [
                    #     max_losses[0].item() < torch.mean(torch.tensor(f1_history))
                    #     and ce_loss_improved,
                    #     max_losses[1].item() < torch.mean(torch.tensor(f2_history))
                    #     and ce_loss_improved,
                    #     max_losses[2].item() < torch.mean(torch.tensor(f3_history))
                    #     and ce_loss_improved,
                    #     max_losses[3].item() < torch.mean(torch.tensor(f4_history))
                    #     and ce_loss_improved,
                    #     max_losses[4].item() < torch.mean(torch.tensor(f5_history))
                    #     and ce_loss_improved,
                    # ]

                    # # img_torch = latents_to_torch_img(latents, self.vae)
                    # # img_torch, nsfw_detected = self.pipe_base.run_safety_checker(
                    # #     img_torch.detach(), device, "cuda"
                    # # )
                    # # img_torch = img_torch[0]
                    # # nsfw_detected = nsfw_detected[0]

                    # loss_vals = [ce_loss] + list(max_losses)
                    # improvements = [ce_loss_improved] + loss_improvements
                    # pred_sequence_seen = tuple(pred_word) in self.previous_target_words

                    # for idx, (loss_val, improved) in enumerate(
                    #     zip(loss_vals, improvements)
                    # ):
                    #     feature_pred = pred_word[idx]
                    #     target_word = target_words[idx]

                    #     if feature_pred == target_word:
                    #         best_loss[idx].append(loss_val.item())
                    #         continue

                    #     if not improved:
                    #         continue

                    #     invalid_feature = any(
                    #         char in feature_pred for char in DISALLOWED_TARGET_CHARS
                    #     )
                    #     if invalid_feature or pred_sequence_seen:
                    #         continue

                    #     history, target_word = change_target(
                    #         loss_val, feature_pred, idx + 1
                    #     )
                    #     # Persist both the refreshed baseline and the new target word.
                    #     best_loss[idx] = history
                    #     target_words[idx] = target_word

                    # target_words = tuple(target_words)
                    # pred_words = tuple(pred_word[:6])

                    # new if for multi_mask

                    target_words = list(target_words)

                    # Since you only have one loss (latent drift), all masks share the same loss history
                    ce_history = best_loss[0] if isinstance(best_loss[0], list) else [best_loss[0]]
                    
                    ce_loss_improved = ce_loss.item() < torch.mean(torch.tensor(ce_history))
                    
                    # All mask positions share the same improvement criterion
                    loss_improvements = [ce_loss_improved] * len(pred_word)
                    
                    pred_sequence_seen = tuple(pred_word) in self.previous_target_words

                    # Update each mask token independently
                    for idx, (improved, feature_pred) in enumerate(zip(loss_improvements, pred_word)):
                        target_word = target_words[idx]

                        # Skip if prediction matches current target
                        if feature_pred == target_word:
                            if isinstance(best_loss[0], list):
                                best_loss[0].append(ce_loss.item())
                            continue

                        if not improved:
                            continue

                        # Check for invalid characters
                        invalid_feature = any(
                            char in feature_pred for char in DISALLOWED_TARGET_CHARS
                        )
                        if invalid_feature or pred_sequence_seen:
                            continue

                        # Update the target word for this position
                        if isinstance(best_loss[0], list):
                            best_loss[0].append(ce_loss.item())
                        else:
                            best_loss[0] = [ce_loss.item()]
                        target_words[idx] = feature_pred
                        
                        # Log the word change
                        words_path = os.path.join(self.out_folder, f"words_mask_{idx}.txt")
                        with open(words_path, "a" if os.path.exists(words_path) else "w") as f:
                            f.write(f"{feature_pred}\n")

                    target_words = tuple(target_words)
                    pred_words = tuple(pred_word[:cfg.num_masks])

                elif cfg.mask_type == "single_mask":
                    pred_token = pred_word[0] if len(pred_word) > 0 else ""
                    invalid_feature = any(
                        char in pred_token for char in DISALLOWED_TARGET_CHARS
                    )

                    all_words_path = os.path.join(self.out_folder, "all_words.txt")
                    with open(
                        all_words_path, "a" if os.path.exists(all_words_path) else "w"
                    ) as f:
                        f.write(f"{pred_token}\n")

                    if pred_token == target_words:
                        best_loss.append(ce_loss.item())
                    elif ce_loss.item() < torch.mean(torch.tensor(best_loss)):
                        if (
                            not invalid_feature
                            and pred_token not in self.previous_target_words
                        ):
                            best_loss = [ce_loss.item()]
                            target_words = pred_token

                            words_path = os.path.join(self.out_folder, "words.txt")
                            with open(
                                words_path, "a" if os.path.exists(words_path) else "w"
                            ) as f:
                                f.write(f"{target_words}\n")

                # if (
                #     cfg.mask_type == "multi_mask"
                #     and predicted_class[0] == cfg.class_to_activate
                # ):
                #     with open(
                #         os.path.join(self.out_folder, f"pred_words.txt"), "a"
                #     ) as f:
                #         f.write(f"{pred_words}\n")

                # Log predicted words for multi_mask (regardless of classifier prediction like above)
                if cfg.mask_type == "multi_mask":
                    with open(
                        os.path.join(self.out_folder, f"pred_words.txt"), "a"
                    ) as f:
                        f.write(f"{pred_words}\n")

                # stupid nsfw check code
                # img_torch = latents_to_torch_img(latents, self.vae)
                # img_torch, nsfw_detected = self.pipe_base.run_safety_checker(
                #     img_torch.detach(), device, "cuda"
                # )
                # img_torch = img_torch[0]
                # nsfw_detected = nsfw_detected[0]

                # this is based on the classifier, which we don't use anymore
                # if (
                #     predicted_class[0] == cfg.class_to_activate
                #     and cfg.mask_type == "single_mask"
                # ):
                #     target_prob_value = torch.nn.functional.softmax(output, dim=1)[
                #         :, cfg.class_to_activate
                #     ]
                #     target_output = output[:, cfg.class_to_activate]
                #     best_words_path = os.path.join(
                #         self.out_folder, "best_words.txt"
                #     )
                #     with open(
                #         best_words_path,
                #         "a" if os.path.exists(best_words_path) else "w",
                #     ) as f:
                #         f.write(
                #             f"{target_words};{target_prob_value.item()};{target_output.item()}\n"
                #         )

                # if (
                #     predicted_class[0] == cfg.class_to_activate
                #     and cfg.mask_type == "multi_mask"
                # ):
                #     saved_prompt = (
                #         f" a picture of a {pred_word[0]} with {pred_word[1]} and {pred_word[2]} "
                #         f"and {pred_word[3]} and {pred_word[4]} and {pred_word[5]}."
                #     )
                #     new_target_prompt = (
                #         f" a picture of a {target_words[0]} with {target_words[1]} and {target_words[2]} "
                #         f"and {target_words[3]} and {target_words[4]} and {target_words[5]}."
                #     )

                #     if os.path.exists(
                #         os.path.join(
                #             self.out_folder, f"{cfg.class_to_activate}_prompt.txt"
                #         )
                #     ):
                #         with open(
                #             os.path.join(
                #                 self.out_folder, f"{cfg.class_to_activate}_prompt.txt"
                #             ),
                #             "a",
                #         ) as f:
                #             f.write(f"Step prompt:  {saved_prompt}")
                #             f.write("\n")
                #             f.write(f"Target prompt: {new_target_prompt}")
                #             f.write("\n")
                #             f.write("*" * 50)
                #             f.write("\n")
                #     else:
                #         with open(
                #             os.path.join(
                #                 self.out_folder, f"{cfg.class_to_activate}_prompt.txt"
                #             ),
                #             "w",
                #         ) as f:
                #             f.write(f"Step prompt:  {saved_prompt}")
                #             f.write("\n")
                #             f.write(f"Target prompt: {new_target_prompt}")
                #             f.write("\n")
                #             f.write("*" * 50)
                #             f.write("\n")

                # Log prompts for multi_mask (regardless of classifier prediction like above)
                if cfg.mask_type == "multi_mask":
                    # Build prompts from the predicted and target words
                    saved_prompt = (
                        f"a picture of a {pred_word[0]} with {pred_word[1]} and {pred_word[2]} "
                        f"and {pred_word[3]} and {pred_word[4]} and {pred_word[5]}"
                    )
                    new_target_prompt = (
                        f"a picture of a {target_words[0]} with {target_words[1]} and {target_words[2]} "
                        f"and {target_words[3]} and {target_words[4]} and {target_words[5]}"
                    )

                    prompt_path = os.path.join(self.out_folder, "prompts.txt")
                    with open(
                        prompt_path,
                        "a" if os.path.exists(prompt_path) else "w",
                    ) as f:
                        f.write(f"Step {inner_optim_step}:\n")
                        f.write(f"  Current:  {saved_prompt}\n")
                        f.write(f"  Target:   {new_target_prompt}\n")
                        f.write(f"  Loss:     {ce_loss.item():.4f}\n")
                        f.write("*" * 50 + "\n")
                
                # Save images if requested
                if cfg.save_images:
                    img_base = latents_to_pil(latents_base.half(), self.vae.half())
                    img_modified = latents_to_pil(latents_modified.half(), self.vae.half())
                    img_base[0].save(
                        os.path.join(self.frame_folder, f"step_{inner_optim_step}_base.png")
                    )
                    img_modified[0].save(
                        os.path.join(self.frame_folder, f"step_{inner_optim_step}_modified.png")
                    )

                # if len(captions) == 50:
                #     break

                # if predicted_class[0] == cfg.class_to_activate:
                #     convergence += 1

                #     if convergence == 3:
                #         print("Convergence reached at step", inner_optim_step)
                #         break
                # else:
                #     convergence = 0

                # Early stopping based on loss convergence
                if ce_loss.item() < cfg.convergence_threshold if hasattr(cfg, 'convergence_threshold') else 0.001:
                    convergence += 1
                    if convergence >= 3:
                        print(f"Convergence reached at step {inner_optim_step}")
                        break
                else:
                    convergence = 0

                # Break if max captions reached (optional)
                if len(captions) >= 50:
                    break

            # if cfg.save_images and predicted_class[0] == cfg.class_to_activate:
            #     img = latents_to_pil(latents, self.vae)
            #     img[0].save(os.path.join(self.out_folder, "final_img.png"))

            # Save final images for this outer iteration
            if cfg.save_images:
                img_base = latents_to_pil(latents_base.half(), self.vae.half())
                img_modified = latents_to_pil(latents_modified.half(), self.vae.half())
                img_base[0].save(os.path.join(self.out_folder, f"final_iter_{outer_idx}_base.png"))
                img_modified[0].save(os.path.join(self.out_folder, f"final_iter_{outer_idx}_modified.png"))

            # Store the target words for this iteration
            self.previous_target_words.append(target_words)

        return self.out_folder

    def find_unique_masks(self, num_masks_to_find: int) -> str:
        """Run multiple optimization rounds to find unique mask words.
        
        Args:
            num_masks_to_find: Number of unique mask words to discover
            
        Returns:
            Path to the output directory containing all results
        """
        cfg = self.config
        device = self.device
        
        # Create single output directory for all runs
        self.out_folder = ensure_output_dir(
            cfg.outdir, cfg.class_to_activate, cfg.to_kwargs()
        )
        validate_options(cfg.sp_init, cfg.no_match_uniform, cfg.mask_type)
        
        self.frame_folder = os.path.join(self.out_folder, "frames")
        os.makedirs(self.frame_folder, exist_ok=True)
        
        found_words = set()
        run_idx = 0
        successful_runs = 0
        
        self._print_section(f"Finding {num_masks_to_find} unique mask words")
        
        while successful_runs < num_masks_to_find:
            self._print_section(
                f"Run {run_idx + 1} (Found: {successful_runs}/{num_masks_to_find})"
            )
            
            # Set different seed for each run
            diff_seed = cfg.seed + run_idx
            
            # Initialize text encoder and optimizer for this run
            self.text_encoder = TextPipelineWithProjection(
                translation_matrix=self.translation_matrix, 
                n_soft_prompt=self.config.n_soft_prompt,
                clip_checkpoint=self.clip_orig_text_encoder_name,
                clip_subfolder_encoder=self.clip_orig_text_encoder_name_subfolder,
                clip_subfolder_tokenizer=self.clip_orig_tokenizer_name_subfolder,
            )
            self.text_encoder = self.text_encoder.to(device).half()
            self.text_encoder_2 = TextPipelineWithProjection(
                translation_matrix=self.translation_matrix_2,
                n_soft_prompt=self.config.n_soft_prompt,
                clip_checkpoint=self.clip_orig_text_encoder_name_2,
                clip_subfolder_encoder=self.clip_orig_text_encoder_name_subfolder_2,
                clip_subfolder_tokenizer=self.clip_orig_tokenizer_name_subfolder_2,
            )
            self.text_encoder_2 = self.text_encoder_2.to(device).half()

            self.text_encoder_2.bert = self.text_encoder.bert  # same BERT and same soft prompt params

            self.optimizer = torch.optim.AdamW(
                self.text_encoder.prompt_parameters, 
                lr=self.config.lr, 
                eps=1e-4
            )
            
            # Initialize optimization
            dummy_prompt = "X " * cfg.n_tokens
            dummy_prompt = dummy_prompt.strip()
            
            if cfg.prefix not in ["", None]:
                prompts = [cfg.prefix + " " + dummy_prompt]
            else:
                prompts = [dummy_prompt]
            
            dim = cfg.gen_dim
            g = cfg.guidance_scale
            
            convergence = 0
            prompt = getattr(cfg, "prompt", None) or "A picture of a"
            target_words = initialize_dynamic_target(cfg.mask_type)
            best_loss = None
            final_word = None
            
            # Track all words and their losses for this run
            run_words_log = []
            
            # Inner optimization loop
            for inner_optim_step in range(cfg.optim_steps):
                self._print_subsection(f"Step {inner_optim_step + 1}/{cfg.optim_steps}")
                
                label = update_target(target_words, prompt, cfg.mask_type)
                self._print_line("Target prompt:", label)
                
                text, _ = text_enc_with_proj(
                    prompts, 
                    self.tokenizer, 
                    self.clip_orig_text_encoder, 
                    device=device
                )
                text_2, _ = text_enc_with_proj(
                    prompts,
                    self.tokenizer_2,
                    self.clip_orig_text_encoder_2,
                    device=device
                )
                
                cond_1, pooled_cond_1, masked_loss_1, pred_word = self.text_encoder(
                    prompt=prompt,
                    label=label,
                    tau=cfg.tau,
                    mask_type=cfg.mask_type,
                    target_words=target_words,
                )
                cond_2, pooled_cond_2, masked_loss_2, pred_word_2 = self.text_encoder_2(
                    prompt=prompt,
                    label=label,
                    tau=cfg.tau,
                    mask_type=cfg.mask_type,
                    target_words=target_words,
                )
                clip_prompt_embeds = torch.cat([cond_1, cond_2], dim=-1)  # concat CLIP embeddings from both encoders
                t5_prompt_embed = torch.zeros(  # recreate no T5 use
                    (
                        1, # batch_size * num_images_per_prompt,
                        self.pipe_base.tokenizer_max_length,
                        self.pipe_base.transformer.config.joint_attention_dim,
                    ),
                    device=device,
                    dtype=self.pipe_base.dtype,
                )
                clip_prompt_embeds = torch.nn.functional.pad(  # Pad CLIP prompt embeds to match T5 dimension as in the paper
                    clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
                )
                cond = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)  # Final conditioning combines CLIP and T5 (though T5 is not used here)
                pooled_cond = torch.cat([pooled_cond_1, pooled_cond_2], dim=-1)  # Final pooled conditioning also concatenates both encoders
                masked_loss = (masked_loss_1 + masked_loss_2) / 2
                
                self._print_line("Predicted words:", self._format_words(pred_word))
                
                uncond_1, pooled_uncond_1 = text_enc_with_proj(
                    [""],
                    self.tokenizer,
                    self.clip_orig_text_encoder,
                    text.shape[1],
                    device=device,
                )
                uncond_2, pooled_uncond_2 = text_enc_with_proj(
                    [""],
                    self.tokenizer_2,
                    self.clip_orig_text_encoder_2,
                    text_2.shape[1],
                    device=device,
                )
                negative_clip_prompt_embeds = torch.cat([uncond_1, uncond_2], dim=-1)
                t5_negative_prompt_embed = torch.zeros(  # recreate no T5 use
                    (
                        1, # batch_size * num_images_per_prompt,
                        self.pipe_base.tokenizer_max_length,
                        self.pipe_base.transformer.config.joint_attention_dim,
                    ),
                    device=device,
                    dtype=self.pipe_base.dtype,
                )
                negative_clip_prompt_embeds = torch.nn.functional.pad(  # Pad CLIP prompt embeds to match T5 dimension as in the paper
                    negative_clip_prompt_embeds, (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1])
                )
                uncond = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
                pooled_uncond = torch.cat([pooled_uncond_1, pooled_uncond_2], dim=-1)  # Final pooled conditioning also concatenates both encoders

                # Generate images from both pipelines
                #latents_base, latents_modified,kk_base,kk_mod 
                #latents_base, latents_modified
                latents_base, latents_modified,kk_base,kk_mod  = generate_two_branches(
                    self.pipe_base, 
                    self.pipe_modified,
                    cond, 
                    uncond, 
                    pooled_cond,
                    pooled_uncond,
                    cfg, 
                    dim=dim, 
                    guidance_scale=g, 
                    seed=diff_seed,
                )
                

                ce_loss = torch.nn.functional.l1_loss(kk_mod,kk_base.detach())
                # Compute loss (latent drift)
                #ce_loss = torch.nn.functional.l1_loss(
                #    latents_modified, latents_base.detach()
                #)
                if cfg.objective_direction == 'maximize':
                    ce_loss = -ce_loss
                
                # Initialize best_loss on first step
                if inner_optim_step == 0:
                    if cfg.mask_type == "multi_mask":
                        best_loss = [[ce_loss.item()]]
                    else:
                        best_loss = [ce_loss.item()]
                
                # Display losses
                if cfg.mask_type == "multi_mask":
                    self._print_line("Masked loss:", f"{masked_loss.item():.4f}")
                    self._print_line("Class loss:", f"{ce_loss.item():.4f}")
                    best_losses = " | ".join(
                        f"B{idx}={torch.mean(torch.tensor(b_loss)):.4f}"
                        for idx, b_loss in enumerate(best_loss, 1)
                    )
                    self._print_line("Best losses:", best_losses)
                else:
                    self._print_line("Masked loss:", f"{masked_loss.item():.4f}")
                    self._print_line("Class loss:", f"{ce_loss.item():.4f}")
                    self._print_line(
                        "Best loss:", f"{torch.mean(torch.tensor(best_loss)):.4f}"
                    )
                
                loss = ce_loss + masked_loss
                self._print_line("Step summary:", f"loss={loss.item():.4f}")
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update target words based on mask type
                if cfg.mask_type == "multi_mask":
                    target_words = list(target_words)
                    ce_history = (
                        best_loss[0] if isinstance(best_loss[0], list) 
                        else [best_loss[0]]
                    )
                    ce_loss_improved = ce_loss.item() < torch.mean(
                        torch.tensor(ce_history)
                    )
                    loss_improvements = [ce_loss_improved] * len(pred_word)
                    
                    for idx, (improved, feature_pred) in enumerate(
                        zip(loss_improvements, pred_word)
                    ):
                        target_word = target_words[idx]
                        
                        if feature_pred == target_word:
                            if isinstance(best_loss[0], list):
                                best_loss[0].append(ce_loss.item())
                            continue
                        
                        if not improved:
                            continue
                        
                        invalid_feature = any(
                            char in feature_pred for char in DISALLOWED_TARGET_CHARS
                        )
                        if invalid_feature:
                            continue
                        
                        if isinstance(best_loss[0], list):
                            best_loss[0].append(ce_loss.item())
                        else:
                            best_loss[0] = [ce_loss.item()]
                        target_words[idx] = feature_pred
                    
                    target_words = tuple(target_words)
                    # Track the first word as the "final" word for this optimization
                    final_word = target_words[0] if target_words else None
                    
                    # Get current best loss (mean of history)
                    current_best_loss = torch.mean(torch.tensor(best_loss[0])).item()
                    run_words_log.append((str(target_words), current_best_loss))
                    
                elif cfg.mask_type == "single_mask":
                    pred_token = pred_word[0] if len(pred_word) > 0 else ""
                    invalid_feature = any(
                        char in pred_token for char in DISALLOWED_TARGET_CHARS
                    )
                    
                    if pred_token == target_words:
                        best_loss.append(ce_loss.item())
                    elif ce_loss.item() < torch.mean(torch.tensor(best_loss)):
                        if not invalid_feature:
                            best_loss = [ce_loss.item()]
                            target_words = pred_token
                    
                    final_word = target_words
                    
                    # Get current best loss (mean of history)
                    current_best_loss = torch.mean(torch.tensor(best_loss)).item()
                    run_words_log.append((str(target_words), current_best_loss))
                
                # Check for convergence
                convergence_threshold = (
                    cfg.convergence_threshold 
                    if hasattr(cfg, 'convergence_threshold') 
                    else 0.001
                )
                if ce_loss.item() < convergence_threshold:
                    convergence += 1
                    if convergence >= 3:
                        print(f"Convergence reached at step {inner_optim_step}")
                        break
                else:
                    convergence = 0
                
                # Save images if requested
                if cfg.save_images:
                    img_base = latents_to_pil(latents_base.half(), self.vae.half())
                    img_modified = latents_to_pil(latents_modified.half(), self.vae.half())
                    run_frame_folder = os.path.join(self.frame_folder, f"run_{run_idx}")
                    os.makedirs(run_frame_folder, exist_ok=True)
                    img_base[0].save(
                        os.path.join(
                            run_frame_folder, f"step_{inner_optim_step}_base.png"
                        )
                    )
                    img_modified[0].save(
                        os.path.join(
                            run_frame_folder, f"step_{inner_optim_step}_modified.png"
                        )
                    )
            
            # Save all words and losses from this run to words_run_X.txt
            words_path = os.path.join(self.out_folder, f"words_run_{run_idx}.txt")
            with open(words_path, "w") as f:
                for word, loss_val in run_words_log:
                    f.write(f"{word}\t{loss_val:.6f}\n")
            
            # After optimization, check if we found a new word
            if final_word is not None:
                # Convert to string for comparison
                word_str = final_word if isinstance(final_word, str) else str(final_word)
                
                # Check if it's a new word
                if word_str not in found_words:
                    found_words.add(word_str)
                    successful_runs += 1
                    
                    # Get the final loss value
                    final_loss = run_words_log[-1][1] if run_words_log else 0.0
                    
                    self._print_line("✓ New word found:", f"{word_str} (loss: {final_loss:.6f})")
                    
                    # Save final images for this successful run
                    if cfg.save_images:
                        img_base = latents_to_pil(latents_base.half(), self.vae.half())
                        img_modified = latents_to_pil(latents_modified.half(), self.vae.half())
                        img_base[0].save(
                            os.path.join(
                                self.out_folder, 
                                f"final_success_{successful_runs}_base.png"
                            )
                        )
                        img_modified[0].save(
                            os.path.join(
                                self.out_folder, 
                                f"final_success_{successful_runs}_modified.png"
                            )
                        )
                    
                    # Append to found_words.txt with loss
                    found_words_path = os.path.join(self.out_folder, "found_words.txt")
                    with open(
                        found_words_path, 
                        "a" if os.path.exists(found_words_path) else "w"
                    ) as f:
                        f.write(f"{word_str}\t{final_loss:.6f}\n")
                else:
                    self._print_line("✗ Word already found:", word_str)
            else:
                self._print_line("✗ No final word", "optimization failed")
            
            run_idx += 1
            
            # Safety check to prevent infinite loops
            # if run_idx > num_masks_to_find * 10:
            if run_idx > num_masks_to_find * 2:
                print(
                    f"Warning: Reached maximum attempts ({run_idx}). "
                    f"Found {successful_runs}/{num_masks_to_find} unique words. Stopping."
                )
                break
        
        self._print_section(
            f"Completed: Found {successful_runs} unique words in {run_idx} runs"
        )
        
        # Print summary
        print("\nFound words:")
        for word in sorted(found_words):
            print(f"  - {word}")
        
        return self.out_folder
