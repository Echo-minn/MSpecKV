import torch
from typing import Optional, List, Tuple
from tqdm import tqdm
from utils.quant_kv import PastKeyValues, quantize_past_kv, dequantize_past_kv

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"  

DATASET_ID     = "Salesforce/wikitext"
DATASET_SUBSET = "wikitext-2-v1"
PROMPT_PATH = "./sample.prompt"

DECODE_LEN      = 256
PROMPT_MAX_LEN  = 4096
TOKEN_BUDGET    = 1024
DRAFT_AHEAD_LEN = 5


def check_kv_compatibility(past_tgt: PastKeyValues, past_draft: PastKeyValues) -> bool:
    """
    Check if target and draft KV caches have compatible shapes.
    They must have the same number of layers, heads, and head dimensions.
    """
    if len(past_tgt) != len(past_draft):
        return False
    
    for (K_tgt, V_tgt), (K_draft, V_draft) in zip(past_tgt, past_draft):
        # Check shape compatibility: [batch, heads, seq, head_dim]
        if K_tgt.shape[1] != K_draft.shape[1]:  # num_heads
            return False
        if K_tgt.shape[3] != K_draft.shape[3]:  # head_dim
            return False
        if V_tgt.shape[1] != V_draft.shape[1]:  # num_heads
            return False
        if V_tgt.shape[3] != V_draft.shape[3]:  # head_dim
            return False
    
    return True


def extract_kv_delta(old_kv: Optional[PastKeyValues], new_kv: PastKeyValues) -> PastKeyValues:
    """
    Extract only the new KV rows (last position) from new_kv.
    If old_kv is None, return new_kv (first forward pass).
    """
    if old_kv is None:
        return new_kv
    
    delta_kv = []
    for (old_K, old_V), (new_K, new_V) in zip(old_kv, new_kv):
        # Extract last position: [batch, heads, seq, dim] -> [batch, heads, 1, dim]
        K_delta = new_K[:, :, -1:, :]
        V_delta = new_V[:, :, -1:, :]
        delta_kv.append((K_delta, V_delta))
    return delta_kv


def reconstruct_kv_from_deltas(
    base_kv: PastKeyValues,
    delta_kvs: List[PastKeyValues]
) -> PastKeyValues:
    """
    Reconstruct full KV cache by concatenating base_kv with all deltas.
    base_kv: committed KV cache (list of (K, V) per layer)
    delta_kvs: list of incremental KV deltas (one per token, each is list of (K, V) per layer)
    """
    if len(delta_kvs) == 0:
        return base_kv
    
    num_layers = len(base_kv)
    reconstructed = []
    
    # Get dtype and device from base_kv to ensure consistency
    base_dtype = base_kv[0][0].dtype
    base_device = base_kv[0][0].device
    
    # For each layer
    for layer_idx in range(num_layers):
        base_K, base_V = base_kv[layer_idx]
        K_parts = [base_K]
        V_parts = [base_V]
        
        # Append each delta's corresponding layer
        for delta_kv in delta_kvs:
            delta_K, delta_V = delta_kv[layer_idx]
            # Ensure delta tensors match base dtype and device
            delta_K = delta_K.to(dtype=base_dtype, device=base_device)
            delta_V = delta_V.to(dtype=base_dtype, device=base_device)
            K_parts.append(delta_K)
            V_parts.append(delta_V)
        
        # Concatenate along sequence dimension (dim=2)
        K_full = torch.cat(K_parts, dim=2)
        V_full = torch.cat(V_parts, dim=2)
        reconstructed.append((K_full, V_full))
    
    return reconstructed


@torch.no_grad()
def forward_one_step(
    model,
    input_ids: torch.Tensor,          # [batch, 1] or [batch, seq_len]
    past_key_values: Optional[PastKeyValues] = None,
    use_cache: bool = True
):
    """
    Runs one decoding step with provided KV cache.
    Returns:
        next_logits: [batch, vocab_size]
        new_past_key_values: updated KV cache (append one token)
    """
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=use_cache
    )
    logits = outputs.logits[:, -1, :]  # last token
    new_past = outputs.past_key_values
    return logits, new_past


@torch.no_grad()
def hybrid_quantspec_specache_generate(
    model_target,             # full-precision model
    model_draft,              # 4-bit quantized model
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    draft_ahead_num: int = 4,
    num_bits_kv: int = 8,
    temperature: float = 0.0,  # 0 => greedy
):
    device = next(model_target.parameters()).device
    
    # Get model dtype for ensuring KV cache dtype consistency
    model_dtype = next(model_target.parameters()).dtype

    # 1. Encode prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    batch_size = input_ids.size(0)
    assert batch_size == 1, "for simplicity, single batch"

    # 2. Warm up target model on prompt to build initial full-precision KV
    logits_tgt, past_tgt = forward_one_step(
        model_target,
        input_ids=input_ids,
        past_key_values=None,
        use_cache=True
    )
    # past_tgt now has KV for the full prompt

    # 3. Warm up draft model on prompt to build its KV
    logits_draft, past_draft = forward_one_step(
        model_draft,
        input_ids=input_ids,
        past_key_values=None,
        use_cache=True
    )

    # Check if models have compatible KV cache shapes
    kv_compatible = check_kv_compatibility(past_tgt, past_draft)
    if not kv_compatible:
        print("WARNING: Target and draft models have incompatible KV cache shapes!")
        print("  This means they have different architectures (different number of layers, heads, or head dimensions).")
        print("  KV cache reuse (SpeCache) will be disabled. Only token prediction will be used from draft model.")
        print("  For optimal performance, use a quantized version of the target model as the draft model.")
        use_kv_cache = False
    else:
        use_kv_cache = True

    # Current last token id (for both models)
    last_token_id = input_ids[:, -1]

    # Generated tokens buffer (we'll store only the new ones)
    generated_ids = []

    # Stats
    total_proposed = 0
    total_accepted = 0
    total_target_forwards = 0

    # Track draft KV state for efficient delta extraction
    past_draft_prev = past_draft

    # 4. Main decoding loop with progress bar
    pbar = tqdm(total=max_new_tokens, desc="Generating", unit="token", ncols=100)
    pbar.set_postfix({"accepted": 0, "proposed": 0, "rate": "0.0%"})
    
    while len(generated_ids) < max_new_tokens:
        # ======================================
        # (A) DRAFT SPECULATIVE PASS (QuantSpec)
        # ======================================
        speculative_tokens = []
        speculative_kv_deltas_quantized = []  # list of quantized KV deltas (one per token)

        # Note: we start from current draft KV and last token (aligned with target)
        past_draft_current = past_draft_prev
        for step in range(draft_ahead_num):
            # forward draft one step
            logits_draft, past_draft_new = forward_one_step(
                model_draft,
                input_ids=last_token_id.unsqueeze(0),  # [1,1]
                past_key_values=past_draft_current,
                use_cache=True
            )

            # Extract only the NEW KV delta (last position) if KV cache is compatible
            if use_kv_cache:
                kv_delta = extract_kv_delta(past_draft_current, past_draft_new)
                # Quantize only the delta (efficient!)
                quantized_delta = quantize_past_kv(kv_delta, num_bits=num_bits_kv)
                speculative_kv_deltas_quantized.append(quantized_delta)

            # sample/greedy
            if temperature == 0.0:
                next_id = torch.argmax(logits_draft, dim=-1)  # [1]
            else:
                probs = torch.softmax(logits_draft / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

            speculative_tokens.append(next_id.item())
            total_proposed += 1

            # Update for next iteration
            past_draft_current = past_draft_new
            last_token_id = next_id

            # Early stop if EOS
            if next_id.item() == tokenizer.eos_token_id:
                break

        if len(speculative_tokens) == 0:
            break

        # ==============================================
        # (B) DEQUANT & BUILD SPECULATIVE KV (SpeCache)
        # ==============================================
        if use_kv_cache and len(speculative_kv_deltas_quantized) > 0:
            # Dequantize all deltas and reconstruct speculative KV cache
            # Ensure dequantized tensors match model dtype
            speculative_kv_deltas_fp = [
                dequantize_past_kv(qkv, dtype=model_dtype) for qkv in speculative_kv_deltas_quantized
            ]
            
            # OPTIMIZATION: Build full speculative KV once (not per verification step)
            # Reconstruct full speculative KV by combining committed target KV + speculative deltas
            past_tgt_spec_full = reconstruct_kv_from_deltas(past_tgt, speculative_kv_deltas_fp)
            
            # Ensure all KV tensors have correct dtype and device (do once, not per step)
            past_tgt_spec_full = [
                (K.to(dtype=model_dtype, device=device), V.to(dtype=model_dtype, device=device))
                for K, V in past_tgt_spec_full
            ]
        else:
            # KV cache not compatible - will use target's own KV cache
            past_tgt_spec_full = None

        # ====================================================
        # (C) TARGET VERIFICATION (QuantSpec accept/reject)
        # ====================================================
        # Store committed KV for rollback safety
        past_tgt_committed = past_tgt  # stores KV up to last *accepted* token

        accepted_in_block = 0
        mismatch_happened = False

        # Reset last token for target (align with committed state)
        if len(generated_ids) == 0:
            # last token is prompt's last token
            last_token_tgt = input_ids[:, -1]
        else:
            last_token_tgt = torch.tensor(
                [generated_ids[-1]],
                dtype=torch.long,
                device=device
            )

        # Verify each speculative token
        if use_kv_cache and past_tgt_spec_full is not None:
            # OPTIMIZATION: Use pre-built speculative KV and slice for each verification step
            # Get the committed sequence length to know where to slice
            committed_seq_len = past_tgt_committed[0][0].shape[2]  # [batch, heads, seq_len, dim]
            
            for i, tok_id in enumerate(speculative_tokens):
                # OPTIMIZATION: Slice pre-built speculative KV instead of reconstructing
                # We need KV up to position: committed_len + i + 1 (i+1 speculative tokens)
                target_seq_len = committed_seq_len + i + 1
                
                # Slice the pre-built speculative KV to the right length
                kv_for_this_step = [
                    (K[:, :, :target_seq_len, :], V[:, :, :target_seq_len, :])
                    for K, V in past_tgt_spec_full
                ]
                
                # Run target one step with speculative KV cache (up to position i)
                logits_tgt, past_tgt_new = forward_one_step(
                    model_target,
                    input_ids=last_token_tgt.unsqueeze(0),
                    past_key_values=kv_for_this_step,
                    use_cache=True
                )
                total_target_forwards += 1
                
                # Get target's prediction for this position
                if temperature == 0.0:
                    verified_id = torch.argmax(logits_tgt, dim=-1).item()
                else:
                    probs = torch.softmax(logits_tgt / temperature, dim=-1)
                    verified_id = torch.multinomial(probs, num_samples=1).item()
                
                if verified_id == tok_id:
                    # ACCEPT: commit this token and its KV
                    generated_ids.append(verified_id)
                    total_accepted += 1
                    accepted_in_block += 1

                    # Commit KV: past_tgt now includes this accepted token
                    past_tgt = past_tgt_new
                    past_tgt_committed = past_tgt  # update committed checkpoint

                    last_token_tgt = torch.tensor(
                        [verified_id],
                        dtype=torch.long,
                        device=device
                    )

                    # CRITICAL FIX #1: Sync draft input token with target after acceptance
                    last_token_id = last_token_tgt

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        "accepted": total_accepted,
                        "proposed": total_proposed,
                        "rate": f"{total_accepted/max(total_proposed, 1)*100:.1f}%"
                    })

                    if verified_id == tokenizer.eos_token_id:
                        break
                else:
                    # MISMATCH: rollback & fallback
                    mismatch_happened = True
                    # Rollback to committed KV before speculative block
                    past_tgt = past_tgt_committed

                    # Do a normal target step from last committed token
                    logits_tgt_fallback, past_tgt = forward_one_step(
                        model_target,
                        input_ids=last_token_tgt.unsqueeze(0),
                        past_key_values=past_tgt,
                        use_cache=True
                    )
                    total_target_forwards += 1
                    
                    fallback_id = torch.argmax(logits_tgt_fallback, dim=-1).item()
                    generated_ids.append(fallback_id)

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        "accepted": total_accepted,
                        "proposed": total_proposed,
                        "rate": f"{total_accepted/max(total_proposed, 1)*100:.1f}%"
                    })

                    # Update last_token_tgt
                    last_token_tgt = torch.tensor(
                        [fallback_id],
                        dtype=torch.long,
                        device=device
                    )

                    # CRITICAL FIX #1: Sync draft input token with target after rejection
                    last_token_id = last_token_tgt

                    break  # stop verifying rest of block
        else:
            # KV cache not compatible - use classic speculative decoding (parallel verification)
            # Build input sequence: last committed token + all speculative tokens
            verify_input = torch.cat([
                last_token_tgt.unsqueeze(0),
                torch.tensor(speculative_tokens, device=device, dtype=torch.long).unsqueeze(0)
            ], dim=1)  # [1, 1 + draft_ahead_num]
            
            # Run target model on the full speculative sequence in parallel
            outputs = model_target(
                input_ids=verify_input,
                past_key_values=past_tgt,
                use_cache=True
            )
            logits_tgt = outputs.logits  # [batch, seq_len, vocab_size]
            past_tgt_new = outputs.past_key_values
            total_target_forwards += 1
            
            # Extract logits for each speculative token position
            # logits_tgt[:, 0, :] = prediction after last_token (should match spec_1)
            # logits_tgt[:, 1, :] = prediction after spec_1 (should match spec_2)
            target_logits_per_token = logits_tgt[:, :len(speculative_tokens), :]  # [1, draft_ahead_num, vocab_size]
            
            # Verify each speculative token sequentially using pre-computed logits
            for i, tok_id in enumerate(speculative_tokens):
                # Get target's prediction for this position
                if temperature == 0.0:
                    verified_id = torch.argmax(target_logits_per_token[:, i, :], dim=-1).item()
                else:
                    probs = torch.softmax(target_logits_per_token[:, i, :] / temperature, dim=-1)
                    verified_id = torch.multinomial(probs, num_samples=1).item()
                
                if verified_id == tok_id:
                    # ACCEPT
                    generated_ids.append(verified_id)
                    total_accepted += 1
                    accepted_in_block += 1
                    
                    last_token_tgt = torch.tensor(
                        [verified_id],
                        dtype=torch.long,
                        device=device
                    )
                    last_token_id = last_token_tgt
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        "accepted": total_accepted,
                        "proposed": total_proposed,
                        "rate": f"{total_accepted/max(total_proposed, 1)*100:.1f}%"
                    })
                    
                    if verified_id == tokenizer.eos_token_id:
                        break
                else:
                    # REJECT: Use target's token and stop verifying
                    generated_ids.append(verified_id)
                    mismatch_happened = True
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        "accepted": total_accepted,
                        "proposed": total_proposed,
                        "rate": f"{total_accepted/max(total_proposed, 1)*100:.1f}%"
                    })
                    
                    last_token_tgt = torch.tensor(
                        [verified_id],
                        dtype=torch.long,
                        device=device
                    )
                    last_token_id = last_token_tgt
                    break
            
            # Update target KV cache based on how many tokens were accepted
            if accepted_in_block == len(speculative_tokens):
                # All accepted: need one more target forward to generate next token
                last_accepted_token = torch.tensor(
                    [generated_ids[-1]],
                    dtype=torch.long,
                    device=device
                )
                logits_tgt_final, past_tgt = forward_one_step(
                    model_target,
                    input_ids=last_accepted_token.unsqueeze(0),
                    past_key_values=past_tgt_new,
                    use_cache=True
                )
                total_target_forwards += 1
                
                # Generate next token
                if temperature == 0.0:
                    next_token = torch.argmax(logits_tgt_final, dim=-1).item()
                else:
                    probs = torch.softmax(logits_tgt_final / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                
                generated_ids.append(next_token)
                last_token_id = torch.tensor([next_token], dtype=torch.long, device=device)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    "accepted": total_accepted,
                    "proposed": total_proposed,
                    "rate": f"{total_accepted/max(total_proposed, 1)*100:.1f}%"
                })
                
                if next_token == tokenizer.eos_token_id:
                    break
            else:
                # Some rejected: slice KV to only include accepted tokens + 1 (the rejected token)
                seq_len_to_keep = 1 + accepted_in_block + 1  # last_token + accepted + rejected
                past_tgt = [
                    (K[:, :, :seq_len_to_keep, :], V[:, :, :seq_len_to_keep, :])
                    for K, V in past_tgt_new
                ]
        
        # Update draft KV to match target's committed state
        # CRITICAL FIX #2 & #3: Properly handle draft KV synchronization
        if not mismatch_happened and accepted_in_block == len(speculative_tokens):
            # All tokens accepted: update draft KV to match (draft KV is already correct)
            past_draft_prev = past_draft_current
        else:
            # Some rejected: need to resync draft KV with target's committed state
            # OPTIMIZATION: Only rebuild if we have accepted tokens, otherwise use original prompt KV
            if len(generated_ids) == 0:
                # No tokens generated yet, use original prompt KV
                past_draft_prev = past_draft
            else:
                # OPTIMIZATION: Only rebuild if we have many accepted tokens or if acceptance was very low
                # For small number of accepted tokens, we can use incremental approach
                num_accepted = len(generated_ids)
                
                # If only a few tokens accepted, try to reuse existing draft KV and just rollback
                # Otherwise, rebuild from committed tokens
                if num_accepted <= 10 and accepted_in_block > 0:
                    # OPTIMIZATION: For small sequences, rebuild is acceptable
                    # But we can optimize further by only processing new tokens
                    committed_tokens = torch.cat([
                        input_ids[0],
                        torch.tensor(generated_ids, device=device, dtype=torch.long)
                    ])
                    with torch.no_grad():
                        draft_rebuild_out = model_draft(
                            committed_tokens.unsqueeze(0),
                            use_cache=True,
                            past_key_values=None
                        )
                        past_draft_prev = draft_rebuild_out.past_key_values
                else:
                    # For longer sequences, rebuild from committed tokens
                    # This is expensive but necessary for correctness
                    committed_tokens = torch.cat([
                        input_ids[0],
                        torch.tensor(generated_ids, device=device, dtype=torch.long)
                    ])
                    with torch.no_grad():
                        draft_rebuild_out = model_draft(
                            committed_tokens.unsqueeze(0),
                            use_cache=True,
                            past_key_values=None
                        )
                        past_draft_prev = draft_rebuild_out.past_key_values
            # Note: last_token_id is already synced above (CRITICAL FIX #1)

        # Termination checks
        if len(generated_ids) >= max_new_tokens:
            break
        if len(generated_ids) > 0 and generated_ids[-1] == tokenizer.eos_token_id:
            break

        # If *everything* in the block was accepted and no mismatch:
        # loop continues. If mismatch happened, we already did a fallback step,
        # and next iteration will start from there.

    # Close progress bar
    pbar.close()

    # Done. Decode.
    output_ids = torch.cat([input_ids[0], torch.tensor(generated_ids, device=device)], dim=0)
    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    stats = {
        "total_generated": len(generated_ids),
        "total_proposed": total_proposed,
        "total_accepted": total_accepted,
        "acceptance_rate": total_accepted / max(total_proposed, 1),
        "total_target_forwards": total_target_forwards,
        "target_forwards_per_token": total_target_forwards / max(len(generated_ids), 1),
    }
    return text, stats