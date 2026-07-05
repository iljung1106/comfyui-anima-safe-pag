from types import MethodType

import torch
import torch.nn.functional as F

import comfy.samplers


def _sigma_to_float(sigma):
    if torch.is_tensor(sigma):
        return float(sigma.flatten()[0].item())
    return float(sigma)


def _sigma_active(sigma, sigma_start, sigma_end):
    sigma_start = _sigma_to_float(sigma_start)
    sigma_end = _sigma_to_float(sigma_end)
    if sigma_start < sigma_end:
        sigma_start, sigma_end = sigma_end, sigma_start

    value = _sigma_to_float(sigma)
    return sigma_end <= value <= sigma_start


def _percent_range_to_sigmas(model, start_percent, end_percent):
    start_percent = max(0.0, min(1.0, float(start_percent)))
    end_percent = max(0.0, min(1.0, float(end_percent)))
    if start_percent > end_percent:
        start_percent, end_percent = end_percent, start_percent

    model_sampling = model.get_model_object("model_sampling")
    sigma_start = model_sampling.percent_to_sigma(start_percent)
    sigma_end = model_sampling.percent_to_sigma(end_percent)
    return sigma_start, sigma_end, start_percent, end_percent


def _parse_indices(text, max_index):
    values = set()
    for raw_part in str(text).split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" in part:
            pieces = [p.strip() for p in part.split("-", 1)]
            if len(pieces) != 2 or not pieces[0] or not pieces[1]:
                raise RuntimeError(f"Invalid block range '{part}'. Use values like 18 or 18,20,22.")
            start, end = int(pieces[0]), int(pieces[1])
            if end < start:
                start, end = end, start
            values.update(range(start, end + 1))
        else:
            values.add(int(part))

    indices = sorted(i for i in values if 0 <= i <= max_index)
    if not indices:
        raise RuntimeError(f"No valid block indices found. Valid range is 0 to {max_index}.")
    return indices


def _parse_optional_indices(text, max_index):
    if text is None or not str(text).strip():
        return None
    return _parse_indices(text, max_index)


def _get_anima_blocks(model):
    diffusion_model = model.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        raise RuntimeError("Anima Safe PAG expects an Anima/Cosmos/Predict2-style model with diffusion_model.blocks.")
    return blocks


def _expand_cond_labels(transformer_options, batch_size, device):
    labels = transformer_options.get("anima_safe_pag_cond_or_uncond", None)
    if labels is None:
        labels = transformer_options.get("cond_or_uncond", None)
    if not isinstance(labels, (list, tuple)) or len(labels) == 0:
        return None

    if len(labels) == batch_size:
        expanded = list(labels)
    elif batch_size % len(labels) == 0:
        repeat = batch_size // len(labels)
        expanded = []
        for label in labels:
            expanded.extend([label] * repeat)
    else:
        return None

    return torch.tensor(expanded, device=device)


def _project_attention(attn_module, attn):
    value = attn.reshape(*attn.shape[:-2], attn.shape[-2] * attn.shape[-1])
    if hasattr(attn_module, "output_proj"):
        value = attn_module.output_proj(value)
        if hasattr(attn_module, "output_dropout"):
            value = attn_module.output_dropout(value)
        return value

    if hasattr(attn_module, "o_proj"):
        return attn_module.o_proj(value)

    raise RuntimeError("Unsupported attention module: no output projection found.")


def _sdpa_attention(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)


def _lerp_heads(base, target, strength, heads):
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0:
        return base

    if heads is None:
        return base.lerp(target, strength)

    out = base.clone()
    out[:, :, heads, :] = base[:, :, heads, :].lerp(target[:, :, heads, :], strength)
    return out


def _soft_pag_attention(attn_module, q, k, v, strength, heads):
    normal = _sdpa_attention(q, k, v)
    weak = _lerp_heads(normal, v, strength, heads)
    return _project_attention(attn_module, weak)


def _make_pag_compute_attention(attn_module, original_compute, pag_index, perturbation_strength, head_indices):
    def compute_attention(self, q, k, v, transformer_options=None):
        transformer_options = transformer_options or {}
        labels = _expand_cond_labels(transformer_options, q.shape[0], q.device)
        if labels is None:
            return original_compute(q, k, v, transformer_options=transformer_options)

        pag_mask = labels == pag_index
        if not bool(pag_mask.any()):
            return original_compute(q, k, v, transformer_options=transformer_options)

        perturbed = _soft_pag_attention(self, q, k, v, perturbation_strength, head_indices)
        if bool(pag_mask.all()):
            return perturbed

        normal = original_compute(q, k, v, transformer_options=transformer_options)
        normal[pag_mask] = perturbed[pag_mask]
        return normal

    return MethodType(compute_attention, attn_module)


def _patch_anima_attention(blocks, indices, pag_index, perturbation_strength, head_indices):
    patched = []
    for idx in indices:
        block = blocks[idx]
        attn = getattr(block, "self_attn", None)
        if attn is None or not hasattr(attn, "compute_attention"):
            raise RuntimeError(f"Block {idx} does not expose self_attn.compute_attention.")

        original = attn.compute_attention
        heads = _parse_optional_indices(head_indices, attn.n_heads - 1)
        attn.compute_attention = _make_pag_compute_attention(attn, original, pag_index, perturbation_strength, heads)
        patched.append((attn, original))

    return patched


def _restore_attention(patched):
    for attn, original in patched:
        attn.compute_attention = original


def _rescale_guidance(guidance, cond_pred, cfg_result, rescale, mode):
    rescale = float(rescale)
    if rescale <= 0:
        return guidance

    guidance_result = cfg_result + guidance if mode == "full" else cond_pred + guidance
    reduce_dims = tuple(range(1, guidance_result.ndim))
    std_cond = torch.std(cond_pred, dim=reduce_dims, keepdim=True).clamp_min(1e-6)
    std_guidance = torch.std(guidance_result, dim=reduce_dims, keepdim=True).clamp_min(1e-6)
    factor = std_cond / std_guidance
    factor = rescale * factor + (1.0 - rescale)
    return guidance * factor


class AnimaSafePAG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "block_indices": ("STRING", {"default": "18", "multiline": False}),
                "perturbation_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "head_indices": ("STRING", {"default": "", "multiline": False}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.001}),
                "rescale": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rescale_mode": (["full", "partial"], {"default": "full"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patches"

    def patch(
        self,
        model,
        scale,
        block_indices,
        perturbation_strength,
        head_indices,
        start_percent,
        end_percent,
        rescale,
        rescale_mode,
    ):
        patched_model = model.clone()
        sigma_start, sigma_end, start_percent, end_percent = _percent_range_to_sigmas(
            patched_model,
            start_percent,
            end_percent,
        )
        blocks = _get_anima_blocks(patched_model)
        indices = _parse_indices(block_indices, len(blocks) - 1)

        previous_calc = patched_model.model_options.get("sampler_calc_cond_batch_function", None)
        state = {
            "pred": None,
            "warned_short_output": False,
            "warned_shape": False,
            "printed_active": False,
            "printed_delta": False,
        }

        def default_calc(args):
            return comfy.samplers.calc_cond_batch(
                args["model"],
                args["conds"],
                args["input"],
                args["sigma"],
                args["model_options"],
            )

        def calc_with_previous_if_possible(args, expected_len):
            if previous_calc is None:
                return default_calc(args)

            outputs = previous_calc(args)
            if len(outputs) >= expected_len:
                return outputs

            if not state["warned_short_output"]:
                print(
                    "[Anima Safe PAG] Previous sampler_calc_cond_batch_function returned too few "
                    "predictions for the padded batch. Falling back to ComfyUI calc_cond_batch."
                )
                state["warned_short_output"] = True
            return default_calc(args)

        def calc_cond_batch_with_pag(args):
            conds = list(args["conds"])
            state["pred"] = None

            if scale == 0 or not conds or conds[0] is None:
                if previous_calc is not None:
                    return previous_calc(args)
                return default_calc(args)

            active = _sigma_active(args["sigma"], sigma_start, sigma_end)
            pag_index = len(conds)
            extended_args = dict(args)
            extended_args["conds"] = conds + [conds[0]]

            model_options = dict(args.get("model_options", {}))
            previous_wrapper = model_options.get("model_function_wrapper", None)

            def pag_model_wrapper(model_function, kwargs):
                kwargs = kwargs.copy()
                true_labels = list(kwargs.get("cond_or_uncond", []))
                kwargs["cond_or_uncond"] = true_labels

                c = kwargs.get("c", {}).copy()
                transformer_options = c.get("transformer_options", {}).copy()
                transformer_options["anima_safe_pag_cond_or_uncond"] = true_labels
                transformer_options["cond_or_uncond"] = true_labels
                c["transformer_options"] = transformer_options
                kwargs["c"] = c

                if previous_wrapper is not None:
                    return previous_wrapper(model_function, kwargs)
                return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])

            model_options["model_function_wrapper"] = pag_model_wrapper
            extended_args["model_options"] = model_options

            if active:
                patched = _patch_anima_attention(blocks, indices, pag_index, perturbation_strength, head_indices)
                try:
                    outputs = calc_with_previous_if_possible(extended_args, pag_index + 1)
                finally:
                    _restore_attention(patched)
            else:
                outputs = calc_with_previous_if_possible(extended_args, pag_index + 1)

            if len(outputs) <= pag_index:
                raise RuntimeError(
                    "Anima Safe PAG did not receive the padded prediction. "
                    "This usually means another sampler_calc_cond_batch_function consumed the extended condition list."
                )

            if active:
                state["pred"] = outputs[pag_index]
                if not state["printed_active"]:
                    print(
                        "[Anima Safe PAG] active: "
                        f"blocks={indices}, strength={float(perturbation_strength):.3f}, "
                        f"heads={'all' if not str(head_indices).strip() else str(head_indices).strip()}, "
                        f"range={float(start_percent):.3f}-{float(end_percent):.3f}, "
                        f"conds={len(conds)}+pag"
                    )
                    state["printed_active"] = True
            return outputs[:len(conds)]

        def post_cfg_function(args):
            cfg_result = args["denoised"]
            if scale == 0 or not _sigma_active(args["sigma"], sigma_start, sigma_end):
                return cfg_result

            pag_pred = state.get("pred", None)
            if pag_pred is None:
                return cfg_result

            cond_pred = args["cond_denoised"]
            if pag_pred.shape != cond_pred.shape:
                if not state["warned_shape"]:
                    print(
                        "[Anima Safe PAG] Skipping this step because the perturbed prediction "
                        f"shape {tuple(pag_pred.shape)} does not match cond shape {tuple(cond_pred.shape)}."
                    )
                    state["warned_shape"] = True
                return cfg_result

            guidance = (cond_pred - pag_pred) * float(scale)
            if not state["printed_delta"]:
                delta = (cond_pred - pag_pred).detach().abs().mean().item()
                print(
                    "[Anima Safe PAG] first-step mean |cond - pag| = "
                    f"{delta:.8f}; scale={float(scale):.3f}"
                )
                state["printed_delta"] = True

            guidance = _rescale_guidance(guidance, cond_pred, cfg_result, float(rescale), rescale_mode)
            return cfg_result + guidance

        patched_model.set_model_sampler_calc_cond_batch_function(calc_cond_batch_with_pag)
        patched_model.set_model_sampler_post_cfg_function(post_cfg_function)
        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "AnimaSafePAG": AnimaSafePAG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaSafePAG": "Anima Safe PAG",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
