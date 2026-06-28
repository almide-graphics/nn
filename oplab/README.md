# oplab — the verified-NN-op flywheel

A small manufacturing line for **parity-verified** Almide NN ops. The bet
(see the Aid-On strategy notes): you can win on *breadth* of NN ops if breadth
is **manufactured + stamped**, not hand-written to chase torch — because the
cost of producing a *verified* op collapses when an LLM writes the op and an
independent float64 oracle proves it.

## Split of labour

- **Proposer** (human/LLM): the op body in `../src/<module>.almd`, plus a couple
  of hand-written *invariant* tests (semantic constraints: `silu(0)=0`, WKV of a
  constant is that constant, …).
- **Verifier** (mechanical, this dir): a numpy float64 reference + a seeded input
  spec → an auto-generated, baked Almide parity test → `almide test`.

The generated parity tests are pure WASM-path tests, so they sidestep the native
codegen bug [almide/almide#739](https://github.com/almide/almide/issues/739)
while still giving an independent parity receipt — the op-level analogue of the
model-level gate in `../tools/compare_logits.py`.

## Use

```bash
# regenerate + verify every recipe
tools/.venv/bin/python oplab/flywheel.py
# or one op
tools/.venv/bin/python oplab/flywheel.py wkv
```

## Add a verified op

1. Write the op in `src/<module>.almd` (+ invariant tests).
2. Add `oplab/recipes/<op>.py`:
   ```python
   NAME, MODULE, CALL = "myop", "mymod", "mymod.myop(x)"
   TOL, SEED = 1e-9, 12345
   def make_inputs(rng): return {"x": ("matrix", rng.standard_normal((4, 3)))}
   def reference(x): return ...   # numpy, returns (T, D)
   ```
3. `python oplab/flywheel.py myop` → generates `oplab/gen/myop_parity.almd` and runs it.

Input kinds: `"matrix"` (2-D → `matrix.from_lists`) or `"vector"` (1-D → list).

## Current line

| op | src | kind | parity |
|---|---|---|---|
| `scan_diag` | `src/scan.almd` | diagonal linear recurrence (SSM/Mamba core) | numpy 1e-9 |
| `wkv` | `src/wkv.almd` | RWKV-4 WKV (attention-free mixing) | numpy 1e-9 |
| `gla_diag` | `src/gla.almd` | diagonal gated linear attention | numpy 1e-9 |
| `retnet_retention` | `src/retnet.almd` | RetNet diagonal retention (fixed γ) | numpy 1e-9 |
| `mamba_abar` | `src/mamba.almd` | Mamba S6 discretization Ā=exp(Δ⊙A) | numpy 1e-9 |
| `hgrn` | `src/hgrn.almd` | hierarchically gated linear RNN (floored gate) | numpy 1e-9 |
| `delta_rule_diag` | `src/delta.almd` | diagonal DeltaNet (error-write) update | numpy 1e-9 |
| `ema` | `src/ema.almd` | per-channel exponential moving average | numpy 1e-9 |
| `ema_sq_rows` | `src/emasq.almd` | exponential moving average of squares (RMSProp/Adam 2nd moment, β=0.9) (T,D) | numpy 1e-9 |
| `median3_filter_rows` | `src/median3filter.almd` | length-3 median filter over time (despiking, sort-free med3) (T,D) | numpy 1e-9 |
| `dilation1d_rows` | `src/dilation1d.almd` | 1-D grayscale dilation: sliding max-3 filter, same-pad (morphology) (T,D) | numpy 1e-9 |
| `erosion1d_rows` | `src/erosion1d.almd` | 1-D grayscale erosion: sliding min-3 filter, same-pad (morphology) (T,D) | numpy 1e-9 |
| `rmsprop_update_rows` | `src/rmspropupdate.almd` | preconditioned gradient g/(√v+ε), RMSProp/Adam step direction (2-input, T,D) | numpy 1e-9 |
| `grad_clip_norm_rows` | `src/gradclipnorm.almd` | per-row gradient clipping by L2 norm g·min(1,c/‖g‖) (T,D) | numpy 1e-9 |
| `cumsum_rows` | `src/cumsum.almd` | prefix sum over time | numpy 1e-9 |
| `cummax_rows` | `src/cummax.almd` | cumulative max over time (T,D) | numpy 1e-9 |
| `cumargmax_rows` | `src/cumargmax.almd` | cumulative argmax over time (running peak position) (T,D) | numpy 1e-9 |
| `decayed_max_rows` | `src/decayedmax.almd` | leaky/decaying running max max(x,0.9·m₋₁), peak-hold envelope (T,D) | numpy 1e-9 |
| `leaky_integrate_rows` | `src/leakyintegrate.almd` | leaky integrator s=x+0.9·s₋₁ (RL eligibility trace) (T,D) | numpy 1e-9 |
| `cumargmin_rows` | `src/cumargmin.almd` | cumulative argmin over time (running trough position) (T,D) | numpy 1e-9 |
| `record_count_rows` | `src/recordcount.almd` | running count of record highs over time (record statistics) (T,D) | numpy 1e-9 |
| `up_run_length_rows` | `src/uprunlength.almd` | length of the current consecutive-increase streak over time (T,D) | numpy 1e-9 |
| `down_run_length_rows` | `src/downrunlength.almd` | length of the current consecutive-decrease streak over time (T,D) | numpy 1e-9 |
| `cummin_rows` | `src/cummin.almd` | cumulative min over time (T,D) | numpy 1e-9 |
| `reverse_cumsum_rows` | `src/revcumsum.almd` | suffix sum over time (T,D) | numpy 1e-9 |
| `abs_cumsum_rows` | `src/abscumsum.almd` | cumulative sum of \|x\| over time (T,D) | numpy 1e-9 |
| `cumcount_pos_rows` | `src/cumcountpos.almd` | running count of positive entries over time, Σ[x>0] (T,D) | numpy 1e-9 |
| `cumprod_rows` | `src/cumprod.almd` | cumulative product over time (T,D) | numpy 1e-9 |
| `cummean_rows` | `src/cummean.almd` | cumulative mean over time (T,D) | numpy 1e-9 |
| `lag1_rows` | `src/lag1.almd` | one-step causal delay / shift (T,D) | numpy 1e-9 |
| `autocorr_lag1_rows` | `src/autocorrlag1.almd` | per-row lag-1 autocorrelation (serial smoothness) ∈ [−1,1] (T,1) | numpy 1e-9 |
| `turning_point_count_rows` | `src/turningpoint.almd` | per-row count of local extrema / direction reversals (roughness) (T,1) | numpy 1e-9 |
| `mean_crossings_rows` | `src/meancrossings.almd` | per-row count of mean crossings (zero-crossing rate of centered signal) (T,1) | numpy 1e-9 |
| `diff_rows` | `src/diffrows.almd` | first difference over time (T,D) | numpy 1e-9 |
| `second_diff_rows` | `src/seconddiff.almd` | discrete second difference x_t−2x_{t-1}+x_{t-2}, acceleration/curvature (T,D) | numpy 1e-9 |
| `cumsum_sq_rows` | `src/cumsumsq.almd` | cumulative sum of squares / energy over time (T,D) | numpy 1e-9 |
| `cummax_abs_rows` | `src/cummaxabs.almd` | running L-infinity envelope over time (T,D) | numpy 1e-9 |
| `cummin_abs_rows` | `src/cumminabs.almd` | running lower |x| envelope over time (T,D) | numpy 1e-9 |
| `center_rows` | `src/centerrows.almd` | mean-centering over time / axis=0 (T,D) | numpy 1e-9 |
| `centered_sign_rows` | `src/centeredsign.almd` | sign of each entry vs its row mean (axis=1), competitive ±1/0 mask (T,D) | numpy 1e-9 |
| `zscore_time_rows` | `src/zscoretime.almd` | per-channel standardization over time (T,D) | numpy 1e-9 |
| `cumsum_abs_diff_rows` | `src/cumabsdiff.almd` | cumulative total variation over time (T,D) | numpy 1e-9 |
| `total_variation_rows` | `src/totalvariation.almd` | per-row total variation Σ|x_{j+1}−x_j| (L1 path length, axis=1) (T,1) | numpy 1e-9 |
| `dirichlet_energy_rows` | `src/dirichletenergy.almd` | per-row Dirichlet energy Σ(x_{j+1}−x_j)² (L2 smoothness, axis=1) (T,1) | numpy 1e-9 |
| `cumsum_sign_change_rows` | `src/cumsignchange.almd` | running zero-crossing count: cumulative sign-flips over time (T,D) | numpy 1e-9 |
| `softmax_time_rows` | `src/softmaxtime.almd` | softmax over the time axis / axis=0 (T,D) | numpy 1e-9 |
| `minmax_time_rows` | `src/minmaxtime.almd` | min-max normalization over time / axis=0 (T,D) | numpy 1e-9 |
| `minmax_rows` | `src/minmaxrows.almd` | row-internal min-max normalization to [0,1] / axis=1 (T,D) | numpy 1e-9 |
| `rms_time_rows` | `src/rmstime.almd` | per-channel RMS normalization over time (T,D) | numpy 1e-9 |
| `cummean_abs_rows` | `src/cummeanabs.almd` | cumulative mean of |x| over time (T,D) | numpy 1e-9 |
| `logsumexp_time_rows` | `src/logsumexptime.almd` | log-softmax over the time axis (T,D) | numpy 1e-9 |
| `cumulative_logsumexp_rows` | `src/cumlogsumexp.almd` | cumulative log-sum-exp over time (online softmax log-partition) (T,D) | numpy 1e-9 |
| `logaddexp_rows` | `src/logaddexp.almd` | elementwise log-semiring add log(e^a+e^b), CRF/HMM forward (2-input, T,D) | numpy 1e-9 |
| `demean_sq_rows` | `src/demeansq.almd` | squared deviation from the per-channel time mean (T,D) | numpy 1e-9 |
| `cumsum_clip0_rows` | `src/cumsumclip0.almd` | cumulative sum of the positive part over time (T,D) | numpy 1e-9 |
| `cumsum_clip_neg_rows` | `src/cumsumclipneg.almd` | cumulative sum of the negative part over time (T,D) | numpy 1e-9 |
| `cumvar_rows` | `src/cumvar.almd` | expanding-window population variance over time (T,D) | numpy 1e-9 |
| `cummax_minus_x_rows` | `src/cummaxminusx.almd` | drawdown from the running maximum over time (T,D) | numpy 1e-9 |
| `ternary_twn_rows` | `src/ternarytwn.almd` | TWN per-row ternary quantization to {−1,0,1} (T,D) | numpy 1e-9 |
| `bitnet_sign_rows` | `src/bitnetsign.almd` | BitNet b1 per-element binarization to {−1,+1} (T,D) | numpy 1e-9 |
| `bitnet_absmean_rows` | `src/bitnetabsmean.almd` | BitNet b1.58 per-tensor absmean ternary quantization to {−1,0,1} (T,D) | numpy 1e-9 |
| `ternary_scaled_rows` | `src/ternaryscaled.almd` | TWN scaled/dequantized ternary α·{−1,0,1} per row (T,D) | numpy 1e-9 |
| `cumsum_sign_rows` | `src/cumsumsign.almd` | cumulative sum of sign over time, running ±-vote (T,D) | numpy 1e-9 |
| `x_minus_cummin_rows` | `src/xminuscummin.almd` | rise above the running minimum over time (T,D) | numpy 1e-9 |
| `cumstd_rows` | `src/cumstd.almd` | expanding-window population standard deviation over time (T,D) | numpy 1e-9 |
| `cumrms_rows` | `src/cumrms.almd` | cumulative (uncentered) RMS over time, streaming RMSNorm scale (T,D) | numpy 1e-9 |
| `bitnet_quant_scaled_rows` | `src/bitnetquantscaled.almd` | BitNet b1 binary weight times per-tensor absmean dequant scale β·sign⁺(x) (T,D) | numpy 1e-9 |
| `bitnet_b158_scaled_rows` | `src/bitnetb158.almd` | BitNet b1.58 dequantized weight γ·ternary, per-tensor absmean γ, out ∈ {−γ,0,+γ} (T,D) | numpy 1e-9 |
| `signed_sqrt_rows` | `src/signedsqrt.almd` | signed square root sign(x)·sqrt(\|x\|), odd variance-stabilizing map (T,D) | numpy 1e-9 |
| `ternary_mask_rows` | `src/ternarymask.almd` | nonzero support mask of TWN ternary quant, \|x\|>0.7·E\|x\| per row → {0,1} (T,D) | numpy 1e-9 |
| `sign_match_rows` | `src/signmatch.almd` | elementwise sign agreement of two matrices sign(a)=sign(b) → {0,1} (T,D) | numpy 1e-9 |
| `cumabsmax_minus_abs_rows` | `src/cumabsmaxminusabs.almd` | drawdown from the running magnitude peak, cummax\|x\|−\|x\| over time (T,D) | numpy 1e-9 |
| `hadamard_sign_rows` | `src/hadamardsign.almd` | elementwise sign product sign(a)·sign(b) ∈ {−1,0,1}, bipolar XNOR (T,D) | numpy 1e-9 |
| `hadamard_ternary_rows` | `src/hadamardternary.almd` | elementwise TWN ternary-code product code(a)·code(b) ∈ {−1,0,1} (2-input, T,D) | numpy 1e-9 |
| `binary_dot_rows` | `src/binarydot.almd` | row-wise binary inner product Σ sign(a)·sign(b), the BitNet popcount matmul primitive (T,1) | numpy 1e-9 |
| `ternary_dot_rows` | `src/ternarydot.almd` | row-wise TWN ternary inner product Σ code(a)·code(b), dead-zone aware (2-input, T,1) | numpy 1e-9 |
| `ternary_cosine_rows` | `src/ternarycosine.almd` | row-wise cosine similarity of TWN ternary codes ∈ [−1,1] (2-input, T,1) | numpy 1e-9 |
| `quant_error_rows` | `src/quanterror.almd` | residual of BitNet b1 quantization x − β·sign⁺(x), β per-tensor absmean (T,D) | numpy 1e-9 |
| `ternary_scale_rows` | `src/ternaryscale.almd` | per-row TWN scale factor α = mean \|x\| over kept entries (T,1) | numpy 1e-9 |
| `signed_log1p_rows` | `src/signedlog1p.almd` | sign-preserving log compression sign(x)·log(1+\|x\|), odd RL reward rescale (T,D) | numpy 1e-9 |
| `signed_square_rows` | `src/signedsquare.almd` | sign-preserving square x·\|x\|, odd expansive map (inverse of signed_sqrt) (T,D) | numpy 1e-9 |
| `hamming_sign_rows` | `src/hammingsign.almd` | row-wise binary Hamming distance Σ[sign(a)≠sign(b)], dual of binary_dot (T,1) | numpy 1e-9 |
| `ternary_density_rows` | `src/ternarydensity.almd` | TWN kept-weight density (1−sparsity), mean of ternary mask per row (T,1) | numpy 1e-9 |
| `bitnet_b158_density_rows` | `src/bitnetb158density.almd` | BitNet b1.58 kept-weight density per row under the per-tensor 0.5γ cut (T,1) | numpy 1e-9 |
| `ternary_balance_rows` | `src/ternarybalance.almd` | net polarity of TWN ternary code (#pos−#neg)/D per row ∈ [−1,1] (T,1) | numpy 1e-9 |
| `ternary_code_entropy_rows` | `src/ternarycodeentropy.almd` | Shannon entropy of the TWN ternary code distribution {−1,0,+1} per row (T,1) | numpy 1e-9 |
| `signed_cube_root_rows` | `src/signedcuberoot.almd` | signed cube root sign(x)·\|x\|^(1/3), strong odd variance-stabilizer (T,D) | numpy 1e-9 |
| `quant_error_l2_rows` | `src/quanterrorl2.almd` | per-row L2 norm of BitNet b1 quantization residual ‖x−β·sign⁺‖ (T,1) | numpy 1e-9 |
| `ternary_quant_error_l2_rows` | `src/ternaryquanterrorl2.almd` | per-row L2 norm of TWN ternary residual ‖x−α·ternary‖ (T,1) | numpy 1e-9 |
| `quant_snr_rows` | `src/quantsnr.almd` | per-row signal-to-quantization-noise power ratio ‖x‖²/‖x−β·sign⁺‖² (T,1) | numpy 1e-9 |
| `ternary_quant_snr_rows` | `src/ternaryquantsnr.almd` | per-row SQNR of TWN ternary ‖x‖²/‖x−α·ternary‖² (T,1) | numpy 1e-9 |
| `bitnet_b158_quant_snr_rows` | `src/bitnetb158snr.almd` | per-row SQNR of BitNet b1.58 ‖x‖²/‖x−γ·ternary‖², per-tensor γ (T,1) | numpy 1e-9 |
| `sign_match_rate_rows` | `src/signmatchrate.almd` | per-row fraction of agreeing signs (1/D)Σ[sign(a)=sign(b)] (T,1) | numpy 1e-9 |
| `ternary_match_rate_rows` | `src/ternarymatchrate.almd` | per-row fraction of agreeing TWN ternary codes, dead-zone aware (2-input, T,1) | numpy 1e-9 |
| `x_minus_cummean_rows` | `src/xminuscummean.almd` | online innovation x − running mean over time (T,D) | numpy 1e-9 |
| `cummax_minus_cummin_rows` | `src/cummaxminuscummin.almd` | running peak-to-trough range over time (T,D) | numpy 1e-9 |
| `ternary_quant_error_rows` | `src/ternaryquanterror.almd` | residual of TWN ternary quantization x − α·ternary, the ternary quant noise (T,D) | numpy 1e-9 |
| `bitnet_b158_quant_error_rows` | `src/bitnetb158err.almd` | residual of BitNet b1.58 quantization x − γ·ternary, per-tensor absmean γ (T,D) | numpy 1e-9 |
| `cumprod_sign_rows` | `src/cumprodsign.almd` | running product of signs over time (sign parity tracker) (T,D) | numpy 1e-9 |
| `mean_sign_rows` | `src/meansign.almd` | per-row mean of signs (sign balance / polarity ∈ [−1,1]) (T,1) | numpy 1e-9 |
| `sign_entropy_rows` | `src/signentropy.almd` | per-row binary entropy of the sign distribution H(p₊) (T,1) | numpy 1e-9 |
| `majority_sign_rows` | `src/majoritysign.almd` | majority-vote direction sign(Σ x) per row ∈ {−1,0,1} (T,1) | numpy 1e-9 |
| `sign_change_rows` | `src/signchange.almd` | zero-crossing indicator: 1 where sign flips vs previous timestep (T,D) | numpy 1e-9 |
| `abs_diff_rows` | `src/absdiff.almd` | absolute first difference, per-step total-variation increment (T,D) | numpy 1e-9 |
| `squared_diff_rows` | `src/squareddiff.almd` | squared first difference, per-step change energy (T,D) | numpy 1e-9 |
| `l2_normalize` | `src/l2norm.almd` | row-wise L2 normalization | numpy 1e-9 |
| `cosine_sim_rows` | `src/cosine.almd` | row-wise cosine similarity (T,1) | numpy 1e-9 |
| `dice_coeff_rows` | `src/dice.almd` | row-wise Sørensen–Dice coefficient 2⟨a,b⟩/(‖a‖²+‖b‖²) (2-input, T,1) | numpy 1e-9 |
| `dot_rows` | `src/dotrows.almd` | row-wise dot product (2-input, T,1) | numpy 1e-9 |
| `proj_scalar_rows` | `src/projscalar.almd` | scalar projection of a onto b ⟨a,b⟩/‖b‖ (Gram–Schmidt/PCGrad) (2-input, T,1) | numpy 1e-9 |
| `attention_pool_rows` | `src/attentionpool.almd` | softmax-weighted pool of values by scores Σ softmax(w)·x (2-input, T,1) | numpy 1e-9 |
| `weighted_mean_rows` | `src/weightedmean.almd` | per-row weighted average Σ(w·x)/Σw with explicit weights (2-input, T,1) | numpy 1e-9 |
| `weighted_var_rows` | `src/weightedvar.almd` | per-row weighted variance Σw(x−μ_w)²/Σw with explicit weights (2-input, T,1) | numpy 1e-9 |
| `mahalanobis_diag_rows` | `src/mahalanobis.almd` | diagonal Mahalanobis distance √Σ(a−b)²/v, variance-normalized (3-input, T,1) | numpy 1e-9 |
| `hist_intersection_rows` | `src/histintersect.almd` | histogram-intersection kernel Σ min(a,b), overlap similarity (2-input, T,1) | numpy 1e-9 |
| `ruzicka_similarity_rows` | `src/ruzicka.almd` | Ruzicka / weighted-Jaccard similarity Σmin/Σmax ∈ [0,1] (2-input, T,1) | numpy 1e-9 |
| `softmax_weighted_var_rows` | `src/softmaxwvar.almd` | variance of values under softmax(scores), attention spread (2-input, T,1) | numpy 1e-9 |
| `abs_dot_rows` | `src/absdot.almd` | row-wise absolute dot product (2-input, T,1) | numpy 1e-9 |
| `mean_prod_rows` | `src/meanprod.almd` | row-wise mean product / cross-moment (2-input, T,1) | numpy 1e-9 |
| `max_prod_rows` | `src/maxprod.almd` | row-wise max elementwise product (2-input, T,1) | numpy 1e-9 |
| `min_prod_rows` | `src/minprod.almd` | row-wise min elementwise product (2-input, T,1) | numpy 1e-9 |
| `cosine_dist_rows` | `src/cosinedist.almd` | row-wise cosine distance 1-cos (2-input, T,1) | numpy 1e-9 |
| `l2_dist_rows` | `src/l2dist.almd` | row-wise Euclidean distance (2-input, T,1) | numpy 1e-9 |
| `l1_dist_rows` | `src/l1dist.almd` | row-wise Manhattan distance (2-input, T,1) | numpy 1e-9 |
| `chebyshev_dist_rows` | `src/chebyshev.almd` | row-wise Chebyshev/L∞ distance (2-input, T,1) | numpy 1e-9 |
| `canberra_dist_rows` | `src/canberra.almd` | row-wise Canberra distance Σ\|a−b\|/(\|a\|+\|b\|), coordinate-normalized (2-input, T,1) | numpy 1e-9 |
| `bray_curtis_dist_rows` | `src/braycurtis.almd` | row-wise Bray–Curtis dissimilarity Σ\|a−b\|/Σ(\|a\|+\|b\|), total-normalized (2-input, T,1) | numpy 1e-9 |
| `chi_square_dist_rows` | `src/chisquaredist.almd` | row-wise χ² histogram distance ½Σ(a−b)²/(a+b), per-bin-normalized (2-input, T,1) | numpy 1e-9 |
| `hellinger_dist_rows` | `src/hellinger.almd` | row-wise Hellinger distance √(½Σ(√p−√q)²), amplitude-space L2 (2-input, T,1) | numpy 1e-9 |
| `squared_l2_dist_rows` | `src/sql2dist.almd` | row-wise squared Euclidean distance (2-input, T,1) | numpy 1e-9 |
| `cross_entropy_rows` | `src/celoss.almd` | per-row softmax cross-entropy (T,1) | numpy 1e-9 |
| `soft_cross_entropy_rows` | `src/softcrossentropy.almd` | cross-entropy between two prob distributions −Σp·log q (soft labels) (2-input, T,1) | numpy 1e-9 |
| `focal_loss_rows` | `src/focalloss.almd` | per-row softmax focal loss (γ=2), (1−p)²-modulated CE (2-input, T,1) | numpy 1e-9 |
| `poly1_loss_rows` | `src/poly1loss.almd` | per-row PolyLoss (poly-1) CE + (1−p_t) Taylor term (2-input, T,1) | numpy 1e-9 |
| `kl_div_rows` | `src/kldiv.almd` | per-row KL(p‖q) divergence (2-input, T,1) | numpy 1e-9 |
| `kl_div_logits_rows` | `src/kldivlogits.almd` | stable KL(softmax(a)‖softmax(b)) from logits, distillation (2-input, T,1) | numpy 1e-9 |
| `js_div_rows` | `src/jsdiv.almd` | per-row Jensen-Shannon divergence (2-input, T,1) | numpy 1e-9 |
| `bce_rows` | `src/bce.almd` | per-row binary cross-entropy (2-input, T,1) | numpy 1e-9 |
| `bce_with_logits_rows` | `src/bcelogits.almd` | stable BCE from logits max(z,0)−z·y+log(1+e^−|z|) (2-input, T,1) | numpy 1e-9 |
| `bt_preference_rows` | `src/btpreference.almd` | Bradley–Terry pairwise preference σ(a−b), RLHF reward head (2-input, T,D) | numpy 1e-9 |
| `margin_ranking_loss_rows` | `src/marginranking.almd` | margin/hinge pairwise ranking loss max(0,−y(a−b)+1) (3-input, T,1) | numpy 1e-9 |
| `hinge_rows` | `src/hinge.almd` | per-row SVM hinge loss (2-input, T,1) | numpy 1e-9 |
| `triplet_loss_rows` | `src/tripletloss.almd` | per-row triplet margin loss max(0,‖a−p‖−‖a−n‖+1) (3-input, T,1) | numpy 1e-9 |
| `mse_rows` | `src/mserows.almd` | per-row mean squared error (2-input, T,1) | numpy 1e-9 |
| `rmse_rows` | `src/rmse.almd` | per-row root mean squared error (2-input, T,1) | numpy 1e-9 |
| `mae_rows` | `src/maerows.almd` | per-row mean absolute error (2-input, T,1) | numpy 1e-9 |
| `smooth_l1_rows` | `src/smoothl1.almd` | per-row smooth-L1/Huber loss (2-input, T,1) | numpy 1e-9 |
| `pinball_loss_rows` | `src/pinball.almd` | per-row quantile/pinball loss (asymmetric, τ=0.7) (2-input, T,1) | numpy 1e-9 |
| `eps_insensitive_loss_rows` | `src/epsinsensitive.almd` | per-row ε-insensitive SVR loss max(0,|e|−ε) (2-input, T,1) | numpy 1e-9 |
| `logcosh_loss_rows` | `src/logcoshloss.almd` | per-row log-cosh regression loss (2-input, T,1) | numpy 1e-9 |
| `welsch_loss_rows` | `src/welsch.almd` | per-row Welsch/Leclerc bounded redescending robust loss (2-input, T,1) | numpy 1e-9 |
| `poisson_loss_rows` | `src/poisson.almd` | per-row Poisson NLL (2-input, T,1) | numpy 1e-9 |
| `gaussian_nll_rows` | `src/gaussiannll.almd` | per-row Gaussian NLL 0.5(log σ²+(x−μ)²/σ²), heteroscedastic (3-input, T,1) | numpy 1e-9 |
| `hadamard` | `src/hadamard.almd` | elementwise matrix product (stdlib gap) | numpy 1e-9 |
| `soft_xor_rows` | `src/softxor.almd` | differentiable XOR gate a+b−2ab (DLGN substrate) (2-input, T,D) | numpy 1e-9 |
| `soft_or_rows` | `src/softor.almd` | differentiable OR gate a+b−ab (DLGN substrate) (2-input, T,D) | numpy 1e-9 |
| `soft_and_not_rows` | `src/softandnot.almd` | differentiable inhibition gate A∧¬B a(1−b) (DLGN substrate) (2-input, T,D) | numpy 1e-9 |
| `soft_majority3_rows` | `src/softmajority3.almd` | differentiable 3-input majority gate ab+bc+ca−2abc (threshold logic) (3-input, T,D) | numpy 1e-9 |
| `soft_parity3_rows` | `src/softparity3.almd` | differentiable 3-input parity gate a+b+c−2(ab+bc+ca)+4abc (XOR3, AC⁰-hard) (3-input, T,D) | numpy 1e-9 |
| `soft_consensus3_rows` | `src/softconsensus3.almd` | differentiable 3-input consensus gate abc+(1−a)(1−b)(1−c) (unanimity) (3-input, T,D) | numpy 1e-9 |
| `soft_exactly_one3_rows` | `src/softexactlyone3.almd` | differentiable exactly-one-of-three gate (one-hot/dispatch detector) (3-input, T,D) | numpy 1e-9 |
| `soft_exactly_two3_rows` | `src/softexactlytwo3.almd` | differentiable exactly-two-of-three gate (count-class {2}) (3-input, T,D) | numpy 1e-9 |
| `lerp_rows` | `src/lerp.almd` | elementwise linear interpolation a + t·(b−a) (3-input, T,D) | numpy 1e-9 |
| `fma_rows` | `src/fma.almd` | elementwise fused multiply-add a·b + c (addcmul) (3-input, T,D) | numpy 1e-9 |
| `addcdiv_rows` | `src/addcdiv.almd` | elementwise fused add-and-divide a + b/c (addcdiv, Adam step) (3-input, T,D) | numpy 1e-9 |
| `reglu` | `src/reglu.almd` | ReLU-gated linear unit x⊙max(g,0) | numpy 1e-9 |
| `geglu` | `src/geglu.almd` | GELU-gated linear unit x⊙gelu(g) | numpy 1e-9 |
| `rms_normalize_vec` | `src/rmsnorm.almd` | gain-free row-wise RMS norm | numpy 1e-9 |
| `global_meanpool_rows` | `src/meanpool.almd` | row-wise mean pool (T,1) | numpy 1e-9 |
| `entropy_rows` | `src/entropy.almd` | per-row Shannon entropy (T,1) | numpy 1e-9 |
| `softmax_entropy_rows` | `src/softmaxentropy.almd` | per-row Shannon entropy of softmax(logits), predictive-confidence head (T,1) | numpy 1e-9 |
| `softmax_max_rows` | `src/softmaxmax.almd` | per-row maximum softmax probability (MSP confidence baseline) (T,1) | numpy 1e-9 |
| `soft_argmax_rows` | `src/softargmax.almd` | differentiable argmax: softmax-expected index Σ j·p_j (DSNT) (T,1) | numpy 1e-9 |
| `magnitude_centroid_rows` | `src/magcentroid.almd` | magnitude-weighted index centroid Σ j·|x|/Σ|x| (spectral centroid) (T,1) | numpy 1e-9 |
| `magnitude_spread_rows` | `src/magspread.almd` | magnitude-weighted positional spread √(Σ|x|(j−c)²/Σ|x|) (spectral bandwidth) (T,1) | numpy 1e-9 |
| `spectral_rolloff_rows` | `src/spectralrolloff.almd` | spectral roll-off: index at 85% cumulative |x| energy (T,1) | numpy 1e-9 |
| `logit_margin_rows` | `src/logitmargin.almd` | per-row top1−top2 logit gap (decision margin) (T,1) | numpy 1e-9 |
| `softmax_margin_rows` | `src/softmaxmargin.almd` | per-row top1−top2 softmax probability (margin sampling) (T,1) | numpy 1e-9 |
| `softmax_collision_entropy_rows` | `src/softmaxcollision.almd` | per-row Rényi-2 collision entropy −log Σ softmax(x)² (T,1) | numpy 1e-9 |
| `abs_max_rows` | `src/absmax.almd` | per-row L∞ norm / absmax (T,1) | numpy 1e-9 |
| `range_rows` | `src/rangerows.almd` | per-row range / peak-to-peak (T,1) | numpy 1e-9 |
| `mean_abs_rows` | `src/meanabs.almd` | per-row mean absolute value (T,1) | numpy 1e-9 |
| `signed_geomean_rows` | `src/signedgeomean.almd` | sign-aware geometric mean sign(Πx)·(Π\|x\|)^{1/D} per row (T,1) | numpy 1e-9 |
| `harmonic_mean_abs_rows` | `src/harmonicmeanabs.almd` | harmonic mean of magnitudes D/Σ(1/\|x\|) per row (T,1) | numpy 1e-9 |
| `variance_rows` | `src/variance.almd` | per-row population variance (T,1) | numpy 1e-9 |
| `mean_abs_dev_rows` | `src/meanabsdev.almd` | per-row mean absolute deviation from the mean (L1 dispersion) (T,1) | numpy 1e-9 |
| `gini_coeff_abs_rows` | `src/ginicoeff.almd` | per-row Gini coefficient of magnitudes (concentration/sparsity ∈ [0,1)) (T,1) | numpy 1e-9 |
| `hoyer_sparsity_rows` | `src/hoyersparsity.almd` | per-row Hoyer sparsity (√D−L1/L2)/(√D−1) ∈ [0,1] (T,1) | numpy 1e-9 |
| `magnitude_entropy_rows` | `src/magnitudeentropy.almd` | per-row entropy of the L1-normalized magnitude distribution (energy spread) (T,1) | numpy 1e-9 |
| `participation_ratio_rows` | `src/participationratio.almd` | per-row participation ratio (Σx²)²/Σx⁴, effective dimensionality ∈ [1,D] (T,1) | numpy 1e-9 |
| `std_rows` | `src/stdrows.almd` | per-row population std-dev (T,1) | numpy 1e-9 |
| `skewness_rows` | `src/skewness.almd` | per-row population skewness (3rd standardized moment) m3/σ³ (T,1) | numpy 1e-9 |
| `above_mean_fraction_rows` | `src/abovemeanfraction.almd` | per-row fraction of entries above the mean (nonparametric skew) (T,1) | numpy 1e-9 |
| `kurtosis_rows` | `src/kurtosis.almd` | per-row population excess kurtosis (4th standardized moment) m4/σ⁴−3 (T,1) | numpy 1e-9 |
| `cov_rows` | `src/covrows.almd` | per-row covariance (2-input, T,1) | numpy 1e-9 |
| `corr_rows` | `src/corrrows.almd` | per-row Pearson correlation (2-input, T,1) | numpy 1e-9 |
| `l1_normalize_rows` | `src/l1norm.almd` | row-wise L1 normalization | numpy 1e-9 |
| `l3_norm_rows` | `src/l3norm.almd` | per-row L3 (cubic) norm value (Σ\|x\|³)^{1/3} (T,1) | numpy 1e-9 |
| `rms_rows` | `src/rmsrows.almd` | per-row root-mean-square value sqrt((1/D)Σx²), uncentered (T,1) | numpy 1e-9 |
| `sumpool_rows` | `src/sumpool.almd` | row-wise sum pool (T,1) | numpy 1e-9 |
| `max_rows` | `src/maxrows.almd` | row-wise max pool (T,1) | numpy 1e-9 |
| `max_pool1d_k2_rows` | `src/maxpool1d.almd` | 1-D max pool kernel 2 stride 2, downsampling (T,D)→(T,D/2) | numpy 1e-9 |
| `avg_pool1d_k2_rows` | `src/avgpool1d.almd` | 1-D average pool kernel 2 stride 2, downsampling (T,D)→(T,D/2) | numpy 1e-9 |
| `min_pool1d_k2_rows` | `src/minpool1d.almd` | 1-D min pool kernel 2 stride 2, downsampling (T,D)→(T,D/2) | numpy 1e-9 |
| `upsample_nearest_x2_rows` | `src/upsamplenn.almd` | nearest-neighbour upsampling ×2 (decoder inverse of pooling) (T,D)→(T,2D) | numpy 1e-9 |
| `haar_detail_x2_rows` | `src/haardetail.almd` | level-1 Haar wavelet detail (high-pass) (a−b)/2 (T,D)→(T,D/2) | numpy 1e-9 |
| `min_rows` | `src/minrows.almd` | row-wise min pool (T,1) | numpy 1e-9 |
| `elementwise_max_rows` | `src/emax.almd` | elementwise maximum of two matrices (maxout/skip-max) (2-input, T,D) | numpy 1e-9 |
| `elementwise_min_rows` | `src/emin.almd` | elementwise minimum of two matrices (lower-envelope merge) (2-input, T,D) | numpy 1e-9 |
| `logsumexp_rows` | `src/logsumexp.almd` | stable row-wise log-sum-exp (T,1) | numpy 1e-9 |
| `softmin_rows` | `src/softmin.almd` | row-wise softmin = softmax(-x) | numpy 1e-9 |
| `log_softmax` | `src/logsoftmax.almd` | row-wise stable log-softmax | numpy 1e-9 |
| `logsigmoid` | `src/logsigmoid.almd` | stable log σ(x) | numpy 1e-9 |
| `log_sigmoid_neg` | `src/logsigneg.almd` | log σ(-x) = -softplus(x) | numpy 1e-9 |
| `log1p_abs` | `src/log1pabs.almd` | log(1+|x|) magnitude compression | numpy 1e-9 |
| `log1p_abs_half` | `src/log1pabshalf.almd` | ½·log(1+|x|) softer robust loss | numpy 1e-9 |
| `log_quad` | `src/logquad.almd` | log(1+x²) Cauchy NLL | numpy 1e-9 |
| `log10_quad` | `src/log10quad.almd` | log10(1+x²) decade penalty | numpy 1e-9 |
| `log1p_sq_half` | `src/log1psqhalf.almd` | ½log(1+x²) Cauchy NLL loss | numpy 1e-9 |
| `log2_act` | `src/log2act.almd` | log2(1+|x|) | numpy 1e-9 |
| `log2_quad` | `src/log2quad.almd` | log2(1+x²) | numpy 1e-9 |
| `power_act` | `src/powact.almd` | |x|^1.5 (fractional power) | numpy 1e-9 |
| `log10_act` | `src/log10act.almd` | log10(1+|x|) | numpy 1e-9 |
| `softmax_temperature` | `src/softmaxt.almd` | row-wise temperature softmax(x/τ) | numpy 1e-9 |
| `taylor_softmax_rows` | `src/taylorsoftmax.almd` | row-wise 2nd-order Taylor (polynomial) softmax, exp-free (T,D) | numpy 1e-9 |
| `hardmax_rows` | `src/hardmax.almd` | one-hot indicator of the row maximum (hard attention / top-1 select) (T,D) | numpy 1e-9 |
| `argmax_rows` | `src/argmax.almd` | per-row hard argmax index (classification prediction) (T,1) | numpy 1e-9 |
| `argmin_rows` | `src/argmin.almd` | per-row hard argmin index (nearest-prototype / lowest-cost) (T,1) | numpy 1e-9 |
| `rank_rows` | `src/rankrows.almd` | per-row ordinal rank transform Σ[x_k<x_j], sort-free all-pairs (T,D) | numpy 1e-9 |
| `kendall_tau_rows` | `src/kendalltau.almd` | per-row Kendall τ rank correlation (concordant−discordant, all-pairs) (2-input, T,1) | numpy 1e-9 |
| `median_rows` | `src/medianrows.almd` | per-row true median, sort-free via rank selection (robust center) (T,1) | numpy 1e-9 |
| `hardmin_rows` | `src/hardmin.almd` | one-hot indicator of the row minimum (arg-min / hardest-negative select) (T,D) | numpy 1e-9 |
| `silu` | `src/silu.almd` | SiLU activation | numpy 1e-9 |
| `x_times_sigmoid_neg` | `src/xsigneg.almd` | x·σ(-x) = x/(1+eˣ) reverse SiLU | numpy 1e-9 |
| `cube_sigmoid` | `src/cubesig.almd` | σ(x)³ sharpened gate | numpy 1e-9 |
| `sigmoid_times_x_sq` | `src/sigxsq.almd` | x²·σ(x) gated quadratic | numpy 1e-9 |
| `quad_times_sigmoid_neg` | `src/quadsigneg.almd` | x²·σ(-x) left-gated quadratic | numpy 1e-9 |
| `sigmoid_squared` | `src/sigmoidsq.almd` | σ(x)² delayed gate | numpy 1e-9 |
| `relu_times_sigmoid` | `src/relusig.almd` | max(0,x)·σ(x) one-sided SiLU | numpy 1e-9 |
| `abs_times_sigmoid` | `src/abssig.almd` | |x|·σ(x) gated magnitude | numpy 1e-9 |
| `abs_times_tanh` | `src/abstanh.almd` | |x|·tanh(x) odd signed-square→linear | numpy 1e-9 |
| `softplus` | `src/softplus.almd` | stable softplus | numpy 1e-9 |
| `hardswish` | `src/hardswish.almd` | piecewise-linear swish (MobileNetV3) | numpy 1e-9 |
| `relu6` | `src/relu6.almd` | ReLU clamped at 6 | numpy 1e-9 |
| `relu6_over6` | `src/relu6over6.almd` | ReLU6(x)/6 [0,1] linear gate | numpy 1e-9 |
| `swish_beta` | `src/swishb.almd` | parameterized swish x·σ(βx) | numpy 1e-9 |
| `hardsigmoid` | `src/hardsigmoid.almd` | piecewise-linear sigmoid | numpy 1e-9 |
| `elu` | `src/elu.almd` | exponential linear unit (α scalar) | numpy 1e-9 |
| `mish` | `src/mish.almd` | mish x·tanh(softplus(x)) | numpy 1e-9 |
| `leaky_relu` | `src/leakyrelu.almd` | leaky ReLU (α scalar) | numpy 1e-9 |
| `softsign` | `src/softsign.almd` | softsign x/(1+|x|) | numpy 1e-9 |
| `softsign_beta` | `src/softsignb.almd` | steepness-β softsign x/(1+|βx|) | numpy 1e-9 |
| `cube_softsign` | `src/cubesoftsign.almd` | softsign(x)³ selective odd gate | numpy 1e-9 |
| `half_softsign` | `src/halfsoftsign.almd` | ½·x/(1+|x|) half-amplitude softsign | numpy 1e-9 |
| `softsign_sq` | `src/softsignsq.almd` | softsign(x)² even energy gate | numpy 1e-9 |
| `hardtanh` | `src/hardtanh.almd` | hardtanh clip(x,-1,1) | numpy 1e-9 |
| `clamp_unit` | `src/clampunit.almd` | clamp to [0,1] | numpy 1e-9 |
| `clip_symmetric` | `src/clipsym.almd` | clamp to [-c,c] (c scalar) | numpy 1e-9 |
| `clamp_rows` | `src/clamp.almd` | elementwise clamp to per-element [lo,hi] (tensor bounds) (3-input, T,D) | numpy 1e-9 |
| `sign_act` | `src/signact.almd` | elementwise sign (-1/0/1) | numpy 1e-9 |
| `reciprocal` | `src/reciprocal.almd` | elementwise 1/x | numpy 1e-9 |
| `rsqrt` | `src/rsqrt.almd` | 1/√(|x|+ε) (RMSNorm kernel) | numpy 1e-9 |
| `tanh_act` | `src/tanhact.almd` | tanh activation (via exp) | numpy 1e-9 |
| `tanhshrink` | `src/tanhshrink.almd` | x - tanh(x) | numpy 1e-9 |
| `relu_tanh` | `src/relutanh.almd` | max(0, tanh(x)) one-sided gate | numpy 1e-9 |
| `tanh_squared` | `src/tanhsq.almd` | tanh(x)² even energy gate | numpy 1e-9 |
| `tanh_half` | `src/tanhhalf.almd` | ½·tanh(x) half-amplitude squash | numpy 1e-9 |
| `cube_tanh` | `src/cubetanh.almd` | tanh(x)³ selective odd squash | numpy 1e-9 |
| `threshold` | `src/threshold.almd` | hard threshold (θ scalar) | numpy 1e-9 |
| `softshrink` | `src/softshrink.almd` | soft thresholding / L1 prox (λ scalar) | numpy 1e-9 |
| `hardshrink` | `src/hardshrink.almd` | hard shrinkage (λ scalar) | numpy 1e-9 |
| `softclamp` | `src/softclamp.almd` | clamp to [lo,hi] (2 scalars) | numpy 1e-9 |
| `shifted_softplus` | `src/shiftedsp.almd` | softplus(x)-log2, f(0)=0 (SchNet) | numpy 1e-9 |
| `softplus_beta` | `src/softplusbeta.almd` | temperature softplus (β scalar) | numpy 1e-9 |
| `softplus_inverse` | `src/softplusinv.almd` | log(eˣ-1) inverse softplus (x>0) | numpy 1e-9 |
| `softplus_squared` | `src/softplussq.almd` | softplus(x)² soft one-sided L2 | numpy 1e-9 |
| `softplus_half` | `src/softplushalf.almd` | ½·softplus(x) half-slope smooth ReLU | numpy 1e-9 |
| `log1p_exp_neg` | `src/log1pexpneg.almd` | log(1+e⁻|ˣ|) softplus residual | numpy 1e-9 |
| `scale_shift` | `src/scaleshift.almd` | affine a·x+b (2 scalars) | numpy 1e-9 |
| `celu` | `src/celu.almd` | continuously-differentiable ELU (α scalar) | numpy 1e-9 |
| `gaussian` | `src/gaussian.almd` | Gaussian/RBF activation exp(-x²) | numpy 1e-9 |
| `exp_half_neg_sq` | `src/exphalfnegsq.almd` | exp(-x²/2) unit-σ Gaussian density | numpy 1e-9 |
| `gaussian_scaled` | `src/gaussscaled.almd` | width-σ RBF exp(-x²/2σ²) (scalar) | numpy 1e-9 |
| `gaussian_deriv` | `src/gaussderiv.almd` | -x·exp(-x²) Gaussian derivative | numpy 1e-9 |
| `mexican_hat` | `src/mexicanhat.almd` | (1-x²)exp(-x²/2) Ricker wavelet | numpy 1e-9 |
| `gaussian_times_x` | `src/gaussx.almd` | x·exp(-x²/2) 1st Hermite function | numpy 1e-9 |
| `exp_neg_sq_shift` | `src/expnegsqshift.almd` | exp(-(x-1)²) RBF centered at 1 | numpy 1e-9 |
| `gaussian_bump_shift` | `src/gaussbumpshift.almd` | exp(-(x+1)²) RBF centered at -1 | numpy 1e-9 |
| `exp_neg_quad_shift` | `src/expnegquadshift.almd` | exp(-(x-2)²) RBF centered at 2 | numpy 1e-9 |
| `exp_neg_quad_shift_neg` | `src/expnegquadshiftneg.almd` | exp(-(x+2)²) RBF centered at -2 | numpy 1e-9 |
| `morlet_approx` | `src/morlet.almd` | cos(5x)exp(-x²/2) Morlet wavelet | numpy 1e-9 |
| `sinc` | `src/sinc.almd` | sin(x)/x, sinc(0)=1 | numpy 1e-9 |
| `hard_gaussian` | `src/hardgauss.almd` | max(0,1-x²) Epanechnikov bump | numpy 1e-9 |
| `exp_decay` | `src/expdecay.almd` | exp(-|x|) Laplace kernel | numpy 1e-9 |
| `exp_neg_cube_abs` | `src/expnegcubeabs.almd` | exp(-|x|³) super-Gaussian bump | numpy 1e-9 |
| `exp_neg_abs_cubed_half` | `src/expnegabscubedhalf.almd` | exp(-|x|³/2) gentler super-Gaussian | numpy 1e-9 |
| `exp_neg_quartic` | `src/expnegquartic.almd` | exp(-x⁴) flat-top super-Gaussian | numpy 1e-9 |
| `exp_neg_sixth` | `src/expnegsixth.almd` | exp(-x⁶) higher-order super-Gaussian | numpy 1e-9 |
| `exp_neg_sqrt_abs` | `src/expnegsqrtabs.almd` | exp(-√|x|) heavy-tailed cusp bump | numpy 1e-9 |
| `exp_neg_sqrt2_abs` | `src/expnegsqrt2abs.almd` | exp(-√(2|x|)) √2-scaled cusp bump | numpy 1e-9 |
| `exp_neg_half_abs` | `src/expneghalfabs.almd` | exp(-|x|/2) scale-2 Laplace kernel | numpy 1e-9 |
| `exp_neg_two_sq` | `src/expnegtwosq.almd` | exp(-2x²) narrow Gaussian RBF | numpy 1e-9 |
| `exp_neg_three_sq` | `src/expnegthreesq.almd` | exp(-3x²) tight Gaussian RBF | numpy 1e-9 |
| `exp_neg_four_sq` | `src/expnegfoursq.almd` | exp(-4x²) tightest Gaussian RBF | numpy 1e-9 |
| `logcosh` | `src/logcosh.almd` | log(cosh x) smooth Huber loss | numpy 1e-9 |
| `log_cosh_half` | `src/logcoshhalf.almd` | ½·log(cosh x) half log-cosh loss | numpy 1e-9 |
| `bent_identity` | `src/bentid.almd` | (√(x²+1)-1)/2+x non-saturating | numpy 1e-9 |
| `inv_sqrt1px2` | `src/invsqrt1px2.almd` | x/√(1+x²) algebraic sigmoid | numpy 1e-9 |
| `cauchy_act` | `src/cauchyact.almd` | 1/(1+x²) Lorentzian bump | numpy 1e-9 |
| `inv_quad_shift` | `src/invquadshift.almd` | 1/(1+(x-1)²) Lorentzian at 1 | numpy 1e-9 |
| `recip_1px2_sq` | `src/recip1px2sq.almd` | 1/(1+x²)² squared Lorentzian | numpy 1e-9 |
| `recip_1px2_cube` | `src/recip1px2cube.almd` | 1/(1+x²)³ cubed Lorentzian | numpy 1e-9 |
| `recip_1px4` | `src/recip1px4.almd` | 1/(1+x⁴) flat-top bump | numpy 1e-9 |
| `multiquadric` | `src/multiquadric.almd` | √(1+x²) Hardy multiquadric RBF | numpy 1e-9 |
| `inv_multiquadric` | `src/invmultiquadric.almd` | 1/√(1+x²) inverse multiquadric RBF | numpy 1e-9 |
| `hill_act` | `src/hillact.almd` | x²/(1+x²) Hill saturation gate | numpy 1e-9 |
| `quartic_kernel` | `src/quartic.almd` | (1-x²)₊² Biweight bump | numpy 1e-9 |
| `triangular` | `src/triangular.almd` | max(0,1-|x|) Bartlett window | numpy 1e-9 |
| `tricube` | `src/tricube.almd` | (1-|x|³)₊³ LOWESS tricube kernel | numpy 1e-9 |
| `cosine_kernel` | `src/cosinekernel.almd` | cos(π/2|x|)·1[|x|<1] raised-cosine | numpy 1e-9 |
| `hann_window` | `src/hannwindow.almd` | 0.5(1+cos π|x|)·1[|x|<1] Hann | numpy 1e-9 |
| `smoothstep` | `src/smoothstep.almd` | Hermite 3c²-2c³ S-curve gate | numpy 1e-9 |
| `smootherstep` | `src/smootherstep.almd` | Perlin 6c⁵-15c⁴+10c³ C² S-curve | numpy 1e-9 |
| `isru` | `src/isru.almd` | x/√(1+αx²) inverse sqrt unit (scalar) | numpy 1e-9 |
| `bipolar_sigmoid` | `src/bipolarsig.almd` | 2·σ(x)-1 = tanh(x/2) | numpy 1e-9 |
| `relu_squared` | `src/relusq.almd` | squared ReLU (Primer) | numpy 1e-9 |
| `relu_cubed` | `src/relucubed.almd` | max(0,x)³ one-sided cubic gate | numpy 1e-9 |
| `sqrt_relu` | `src/sqrtrelu.almd` | √(max(0,x)) one-sided sub-linear | numpy 1e-9 |
| `log_relu1p` | `src/logrelu1p.almd` | log(1+relu(x)) one-sided log gate | numpy 1e-9 |
| `squareplus` | `src/squareplus.almd` | algebraic softplus (exp-free) | numpy 1e-9 |
| `square_act` | `src/squareact.almd` | elementwise x² | numpy 1e-9 |
| `half_square` | `src/halfsquare.almd` | ½x² L2 loss kernel | numpy 1e-9 |
| `abs_act` | `src/absact.almd` | elementwise |x| | numpy 1e-9 |
| `neg_abs` | `src/negabs.almd` | -|x| downward V | numpy 1e-9 |
| `cube_act` | `src/cubeact.almd` | elementwise x³ | numpy 1e-9 |
| `quartic_act` | `src/quarticact.almd` | elementwise x⁴ | numpy 1e-9 |
| `quartic_well` | `src/quarticwell.almd` | (x²-1)² double-well potential | numpy 1e-9 |
| `sixth_well` | `src/sixthwell.almd` | (x²-1)³ degree-6 signed well | numpy 1e-9 |
| `quintic_act` | `src/quinticact.almd` | elementwise x⁵ | numpy 1e-9 |
| `sixth_act` | `src/sixthact.almd` | elementwise x⁶ | numpy 1e-9 |
| `sqrt_abs` | `src/sqrtabs.almd` | elementwise √|x| | numpy 1e-9 |
| `sqrt1p_minus1` | `src/sqrt1pm1.almd` | √(1+|x|)-1 sub-linear compression | numpy 1e-9 |
| `abs_smooth` | `src/abssmooth.almd` | √(x²+ε) Charbonnier smooth-ℓ1 | numpy 1e-9 |
| `negate` | `src/negate.almd` | elementwise -x | numpy 1e-9 |
| `exponential_act` | `src/expact.almd` | elementwise exp(x) | numpy 1e-9 |
| `expm1_act` | `src/expm1act.almd` | elementwise exp(x)-1 | numpy 1e-9 |
| `sin_act` | `src/sinact.almd` | elementwise sin(x) (SIREN) | numpy 1e-9 |
| `expsin` | `src/expsin.almd` | exp(sin x) periodic [1/e,e] | numpy 1e-9 |
| `sin_times_x` | `src/sintimesx.almd` | x·sin(x) even modulated periodic | numpy 1e-9 |
| `cos_act` | `src/cosact.almd` | elementwise cos(x) | numpy 1e-9 |
| `cosm1` | `src/cosm1.almd` | cos(x)-1 zero-anchored cosine | numpy 1e-9 |
| `snake` | `src/snake.almd` | x + sin²(x) (periodic) | numpy 1e-9 |
| `cosine_sq` | `src/cosinesq.almd` | cos²(x) periodic [0,1] | numpy 1e-9 |
| `sine_sq` | `src/sinesq.almd` | sin²(x) periodic [0,1] | numpy 1e-9 |
| `cosh_act` | `src/coshact.almd` | hyperbolic cosine (eˣ+e⁻ˣ)/2 | numpy 1e-9 |
| `cosh_minus1` | `src/coshm1.almd` | cosh(x)-1 stiff even L2-like loss | numpy 1e-9 |
| `sinh_act` | `src/sinhact.almd` | hyperbolic sine (eˣ-e⁻ˣ)/2 | numpy 1e-9 |
| `sinh_half` | `src/sinhhalf.almd` | ½·sinh(x) odd anti-saturating | numpy 1e-9 |
| `cosine_decay` | `src/cosdecay.almd` | 0.5(1+cos x) cosine anneal | numpy 1e-9 |
| `sigmoid_act` | `src/sigmoidact.almd` | logistic sigmoid 1/(1+e⁻ˣ) | numpy 1e-9 |
| `logit` | `src/logit.almd` | log(x/(1-x)) inverse sigmoid (0,1) | numpy 1e-9 |
| `gompertz` | `src/gompertz.almd` | exp(-e⁻ˣ) asymmetric sigmoid | numpy 1e-9 |
| `cloglog` | `src/cloglog.almd` | 1-exp(-eˣ) complementary log-log | numpy 1e-9 |
| `sech2` | `src/sech2.almd` | sech²(x)=1-tanh²(x) (tanh derivative) | numpy 1e-9 |

## Not yet automated (the proposer loop)

The op body is still written by hand/LLM. The remaining step is to close the
*neural* half: an LLM proposes the op from the recipe's reference semantics, and
the `almide check`/`test` diagnostics feed back as repair until green. The
verifier built here is what makes that loop safe to run unattended.
