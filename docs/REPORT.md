# Combined Research Report: Crowd Density Classification

Deep Learning for Computer Vision, Phase 2 (April 2026). Authors: Arjit Mathur, Poorav Satija, Garvit Goyal, Divyansh Garg, Laveesh Chanchlani.

This write-up describes what the notebooks in [`notebooks/`](../notebooks) actually do and reports their saved results. All experiments use the same dataset (railway-station images labelled low / medium / high crowd density; see [DATA.md](../DATA.md)), the same stratified split (seed 42: 1,332 train / 285 val / ~285 test) and class-weighted cross-entropy (inverse class frequency). All results are from one run per configuration. With ~285 test images, one image is about 0.35 percentage points (pp).

## Summary

| | Study 1: layer freezing | Study 2: augmentation | Study 3: head ablation |
|---|---|---|---|
| Model | ResNet-50 | ResNet-18 | ViT-B/16 |
| Best result | 93.0% test acc (freeze early layers) | 95.4% clean (Cutout); 89.1% avg. corrupted (Cutout) | 95.1% clean |
| Main finding | Freezing `conv1`-`layer2` beat full fine-tuning by 2.1 pp | No augmentation wins on every corruption | Layer-0 head L0H4 matters most for the texture probe (-22.2 pp) |
| Caveat | High learning rate (Adam, 1e-3) may explain full fine-tuning's instability | Top three augmentations differ by <1 pp on average | Probe subset is only 45 images |

Across all three studies the **medium** class is the hardest: it has the lowest F1 in every clean-test report.

---

## Study 1: Transfer learning with layer freezing (ResNet-50)

**Question.** Which layers of an ImageNet-pretrained ResNet-50 should be frozen when fine-tuning on ~1.3K training images?

**Setup (from the notebook).**

| Code | Strategy | Frozen | Trainable parameters (approx.) |
|---|---|---|---|
| S0 | No freezing | none | 23.5M |
| S1 | Freeze early | `conv1`, `bn1`, `layer1`, `layer2` | 22.1M (`layer3`, `layer4`, `fc`) |
| S2 | Freeze late | `layer3`, `layer4` | 1.4M (`conv1`-`layer2`, `fc`) |
| S3 | Head only | everything except `fc` | 6K |

Parameter counts are derived from the standard ResNet-50 layer sizes with a 3-class head; verify with `sum(p.numel() for p in model.parameters() if p.requires_grad)`.

Training: Adam, learning rate 1e-3, no weight decay, no scheduler; batch size 32; 15 epochs; mixed precision on GPU; the checkpoint with the best validation accuracy is evaluated on the test set. Training augmentation: resize to 224, horizontal flip, ColorJitter (brightness 0.2, contrast 0.2). Validation and test: resize and normalise only.

**Results.**

| Strategy | Best val acc | Test acc | Time (s) | Best epoch | Gap at best epoch | Gap at final epoch |
|---|---|---|---|---|---|---|
| S1 freeze early | 93.3% | **93.0%** | 448.7 | 4 | 1.0 pp | 17.4 pp |
| S0 no freeze | 91.2% | 90.9% | 477.4 | 14 | 6.3 pp | 7.2 pp |
| S2 freeze late | 90.9% | 90.2% | 467.0 | 14 | 4.8 pp | 3.8 pp |
| S3 head only | 88.8% | 88.5% | 431.6 | 14 | 1.2 pp | 1.0 pp |

(Gap = training accuracy minus validation accuracy.)

Per-class F1 on the test set:

| Class | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| Low | 0.929 | 0.956 | 0.929 | 0.896 |
| Medium | 0.847 | 0.886 | 0.866 | 0.817 |
| High | 0.945 | 0.933 | 0.894 | 0.946 |
| Macro | 0.907 | 0.925 | 0.896 | 0.886 |

**Discussion.**

- S1 has the best test accuracy and macro F1. Its advantage over S0 is 2.1 pp (about 6 images), so it is suggestive rather than conclusive with a single seed.
- S1 is best at its best epoch, but not stable: its validation accuracy fell to 78.9% at epoch 12 and 80.0% at epoch 15, which is why its final-epoch gap is 17.4 pp. Selecting the checkpoint on validation accuracy is doing real work here.
- S0's validation accuracy was erratic (55.8% at epoch 4, 62.1% at epoch 11). A learning rate of 1e-3 is aggressive for full fine-tuning of a pretrained network, so this instability may come from the learning rate rather than the number of trainable parameters. Re-running S0 with a lower rate (e.g. 1e-4, with a scheduler) would test this.
- S2 started slowly (68.4% validation accuracy at epoch 1), consistent with the frozen late layers holding ImageNet-specific features.
- S3 reaches 88.5%, so adapting the backbone is worth about 4.5 pp over a linear probe in this setting.
- Practical reading: for a dataset of this size, freezing the first two residual stages is a sensible default that cost no accuracy here and trains fewer parameters.

---

## Study 2: Data augmentation as inductive bias (ResNet-18)

**Question.** Does the augmentation used in training make the classifier invariant to the corresponding test-time corruption?

**Setup (from the notebook).** ImageNet-pretrained ResNet-18, all layers trainable; AdamW, learning rate 1e-4, weight decay 1e-4; batch size 32; 5 epochs; best validation epoch kept. Same model, split and optimiser for every augmentation.

| Augmentation | Training transform |
|---|---|
| Baseline | resize 224, normalise |
| Rotation | RandomRotation up to 25 degrees |
| Crop | resize 256, RandomResizedCrop 224 (scale 0.7-1.0, ratio 0.9-1.1) |
| Color jitter | brightness / contrast / saturation 0.35, hue 0.08 |
| Cutout | RandomErasing on the tensor (area 8-18%, black patch) |

Corrupted copies of the test set (applied at evaluation time, random per image):

| Corruption | Implementation |
|---|---|
| Blur | Gaussian blur, kernel 5, sigma sampled in [1, 2] |
| Noise | additive Gaussian pixel noise, std 0.08, clipped to [0, 1] (named `jpeg_noise` in the code; it is not JPEG compression) |
| Brightness | ColorJitter, brightness up to 0.45, contrast up to 0.2 |
| Occlusion | RandomErasing, area 15-28%, black patch |

One unreadable test image is excluded.

**Results (test accuracy).**

| Augmentation | Clean | Blur | Noise | Brightness | Occlusion | Avg. corrupted | Gap (clean - avg.) |
|---|---|---|---|---|---|---|---|
| Cutout | 95.4% | 89.5% | 77.9% | 95.4% | 93.7% | 89.1% | 6.3 pp |
| Crop | 94.4% | 86.0% | 88.4% | 94.4% | 86.7% | 88.9% | 5.5 pp |
| Color jitter | 95.1% | 87.0% | 80.7% | 95.4% | 90.9% | 88.5% | 6.6 pp |
| Rotation | 92.3% | 82.1% | 90.2% | 92.6% | 87.0% | 88.0% | 4.3 pp |
| Baseline | 94.0% | 86.0% | 78.6% | 94.0% | 82.8% | 85.4% | 8.7 pp |

**Hypotheses versus observations.**

| Augmentation / corruption | Expectation | Observed vs. baseline |
|---|---|---|
| Cutout / occlusion | robust to missing regions | +10.9 pp (93.7% vs 82.8%): clear |
| Color jitter / brightness | robust to photometric change | tied for best (95.4%), +1.4 pp, about 4 images: weak evidence |
| Crop / blur | scale change may help | +0.0 pp: not supported |
| Rotation / noise | not expected | +11.6 pp (90.2% vs 78.6%): unexplained; pixel resampling during rotation is one untested possibility |
| Cutout / noise | not expected | -0.7 pp, the worst noise result |

**Discussion.**

- Cutout has the best clean and average-corrupted accuracy, but its lead over Crop on the average is 0.3 pp and over Color jitter 0.6 pp, which is within noise. Rotation has the smallest clean-to-corrupted gap, partly because its clean accuracy is the lowest.
- The strongest, clearest effect is Cutout against occlusion. The main practical lesson is to choose augmentation according to the corruption expected at deployment (for example, partial occlusion by crowds or pillars, camera noise) rather than to expect one augmentation to cover everything.
- **Failure analysis (occlusion).** For each model, the four highest-confidence errors under occlusion were inspected. For Baseline, Rotation, Crop and Color jitter all four were medium images predicted as low; for Cutout three were medium to low and one was low to medium. This is a sample of four per model, not the full error distribution, but it is consistent with medium images being the most ambiguous class.

---

## Study 3: Attention-head ablation (ViT-B/16)

**Question.** Which attention heads of a fine-tuned ViT matter for texture-, shape- and position-related cues?

**Setup (from the notebook).** `google/vit-base-patch16-224` (12 layers x 12 heads = 144 heads), new 3-class head, fine-tuned for 5 epochs (AdamW, learning rate 2e-5, weight decay 1e-4, batch size 8, best validation epoch kept).

**Ablation.** The self-attention forward pass is patched so that, for one chosen head, the softmax attention probabilities are replaced with a uniform distribution (1/197 over the 196 patch tokens and the [CLS] token). Queries, keys, values and projections are left unchanged, so the head outputs the mean of its value vectors. One head is ablated at a time (144 evaluations).

**Probes.**

| Probe | Input | Intent |
|---|---|---|
| Clean | original image | ordinary classification |
| Texture | deterministic patch shuffle: the order of the 16 px patches inside each 2x2 block is reversed | keeps local texture, destroys global layout |
| Shape | binary edge map (blur, edge filter, threshold) | removes texture and colour, keeps outlines |
| Spatial | occlude each cell of a 4x4 grid, find the cell whose occlusion lowers the true-class logit most, and check whether it falls in the same image quadrant as the quadrant with the highest edge density | whether evidence comes from the crowded region (the quadrant target is a heuristic, not an annotation) |

**Baseline results (full test set, before ablation).**

| Probe | Accuracy | Macro F1 |
|---|---|---|
| Clean | 95.1% | 0.950 |
| Texture (patch shuffle) | 92.6% | 0.928 |
| Shape (edge map) | 44.6% | 0.339 |
| Spatial alignment | 45.8% | n/a (chance is 25%) |

**Head sweep.** The sweep uses a 45-image balanced subset of the test set (15 per class); the spatial score is computed on up to 24 images, 22 of which were classified correctly and so counted (the 22 is inferred from the reported scores, 6/22 and 3/22).

| Probe | Most sensitive head | Accuracy drop | Clean drop for the same head |
|---|---|---|---|
| Texture | L0 H4 | 22.2 pp (10 of 45 images) | 2.2 pp |
| Spatial | L0 H1 | 13.6 pp (6 to 3 of 22 samples) | 0.0 pp |
| Shape | L4 H7 | 4.4 pp (2 of 45 images) | 0.0 pp |

Layer 0 contains 5 of the top-10 heads for the texture probe and 4 of the top-10 for the spatial probe. The top shape-probe drops are all at most 4.4 pp and are spread across layers.

**Discussion.**

- The fine-tuned ViT is almost unaffected by patch shuffling (95.1% to 92.6%), so it does not depend much on global layout. On edge maps it does no better than always predicting the majority class (44.6% vs. 46% for "low"), but that mainly shows it cannot handle a very different input distribution; it was never trained on edge images.
- Removing L0 H4 lowers texture-probe accuracy by 22.2 pp while clean accuracy drops by 2.2 pp, which is clearly above the noise of the probe set. The effects for the shape and spatial heads (2 and 3 samples) are too small to interpret on their own.
- The sweep does not establish a layer-by-layer specialisation (early / middle / late); that would need larger probe sets, repeated seeds and multi-head ablations.
- Worth testing next: patch-shuffle or other shape-forcing augmentation during fine-tuning, a larger probe set, and a pretrained model whose pretraining encourages shape sensitivity.

---

## Limitations

Single seed and split; small test set (~285 images); small ablation probe sets (45 and 22); heuristic spatial ground truth; random test-time corruptions; one dataset from a single source; the learning rates and epoch counts were not tuned per configuration.
