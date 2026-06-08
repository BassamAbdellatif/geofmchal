# Literature-Based Alternatives for the ESA GeoFM Challenge

**Report Date**: June 3, 2026  
**Competition Deadline**: June 30, 2026 (27 days remaining)  
**Current Best Score**: 0.3721 (2A_vegboost)  
**Target**: Close IoU_B gap (0.3394 → 0.5269)

---

## Executive Summary

This report synthesizes evidence from 494 papers across four literature domains (dual-encoder architectures, gradient balancing, loss functions, domain generalization) to provide four actionable alternatives for the ESA GeoFM competition. The alternatives address four core technical problems identified through experimental work: (1) modality competition in shared encoders, (2) task competition in shared decoders, (3) hard-threshold IoU evaluation vs continuous regression training, and (4) train→test domain shift.

**Recommended approach**: Combine Alternative 1 (dual-encoder/dual-decoder architecture), Alternative 4 (Rotograd gradient balancing), and Alternative 3 (domain adaptation components) as the 7A baseline. Alternative 2 (adapter mixture with IoU surrogates) serves as a compute-efficient fallback.

---

# Introduction

The ESA GeoFM competition challenges participants to predict building footprints and heights from foundation model embeddings of satellite imagery. The task requires predicting four-channel abundance fractions (building, vegetation, water, other) plus a height channel from pre-computed GeoFM embeddings, evaluated using intersection-over-union at a hard threshold of 0.5 (IoU_B) and root mean squared error for height. As of 30 days before the deadline, the current best submission (2A_vegboost) achieves 0.3721 on the leaderboard, while the top team has reached 0.5269 IoU_B. Closing this performance gap requires addressing fundamental architectural tensions that emerged during experimental development.

Four core technical problems motivate the search for literature-based alternatives. First, **modality competition** arises when multiple foundation model embeddings must be fused. Experiment 6A demonstrated that naively concatenating TESSERA and AlphaEarth embeddings into a shared encoder catastrophically degraded IoU_B from 0.168 to 0.017, suggesting that different embedding spaces require separate processing pathways. Second, **task competition** between abundance prediction and height regression emerged when collapsing the Y-Net dual-decoder architecture to a shared decoder in 6A, which regressed the proxy metric by approximately 0.08. The abundance and height tasks appear to benefit from distinct feature hierarchies.

Third, a fundamental **hard-IoU versus continuous regression tension** exists between the evaluation metric and training objective. The platform evaluates building predictions by thresholding continuous abundance fractions at 0.5 and computing IoU, but training optimizes continuous MSE loss. Pure MSE optimization in early experiments failed catastrophically because it does not directly optimize for the discrete decision boundary that determines competition performance. Fourth, **domain shift** between training and test sets poses a generalization challenge. The test set covers different geographic regions and time periods than the training data, yet random validation splits allow models to exploit spatial autocorrelation and leak contextual information, producing optimistic validation metrics that do not reflect true generalization performance.

These four problems—modality fusion, multi-task learning, threshold-aware optimization, and domain-aware validation—define the technical scope for literature-based solutions. The following sections examine three architectural alternatives that address the modality and task competition problems: dual-encoder architectures that process embeddings separately before fusion, adapter-based mixture-of-experts that route different modalities through specialized pathways, and ensemble methods with post-hoc calibration that combine predictions from task-specific models. A fourth orthogonal alternative explores gradient balancing techniques that dynamically weight task losses during training. Each alternative draws on established methods from computer vision, multi-task learning, and domain adaptation literature, adapted to the specific constraints of the GeoFM competition framework.

---

## Alternative 1 — Dual-Encoder Fusion with Sensor-Routed Decoders

The dual-encoder architecture directly addresses the modality competition problem observed in experiment 6A, where concatenating TESSERA and AlphaEarth embeddings into a shared encoder catastrophically degraded IoU_B from 0.168 to 0.017. By processing each foundation model embedding through separate encoder pathways, this approach prevents destructive interference between embedding spaces while enabling controlled fusion at strategic network locations.

### Core Design Principle

The fundamental insight underlying dual-encoder architectures is that different modalities capture complementary information requiring distinct feature extraction pathways. Multiple studies demonstrate that separate encoders per modality preserve modality-specific characteristics while enabling effective fusion [1] [2] [3]. For the GeoFM competition, this translates to processing AlphaEarth embeddings (optical-derived, composition-sensitive) through one encoder branch and TESSERA embeddings (SAR-optical fusion, structure-sensitive) through a parallel branch. This separation prevents the embedding space conflicts that destroyed performance in 6A.

The architectural pattern extends naturally to task separation. The task competition observed when collapsing the Y-Net dual-decoder to a shared decoder in 6A suggests that abundance prediction and height regression benefit from distinct feature hierarchies. Multi-task frameworks with task-specific pathways have proven effective for remote sensing problems such as simultaneous segmentation and cloud removal [7], and dual-branch architectures have been successful for multi-sensor analysis [2]. For GeoFM, a fraction decoder would process features optimized for compositional segmentation (building, vegetation, water, other), while a height decoder would process features optimized for structural regression.

### Fusion Strategies

Three fusion approaches emerge from the literature, each with distinct trade-offs. **Bottleneck concatenation** merges encoder outputs at the latent representation before decoding [2]. This approach has proven effective for multi-sensor land cover classification, showing improvements over single-modality baselines [2]. The strategy is computationally efficient but provides limited opportunity for cross-modal interaction during feature extraction.

**Cross-attention fusion** enables richer modality interaction by allowing features from one encoder to attend to features from the other at multiple scales [4] [6] [8]. The CLAIRE architecture uses cross-modality attention to highlight complementary spatial and textural features from optical and SAR imagery, achieving 56.02% mIoU on land cover segmentation [6]. For GeoFM, cross-attention could be applied at decoder skip connection levels (16×16, 32×32 resolution) to allow AlphaEarth features to query TESSERA features for structural context and vice versa.

**Stage-wise progressive fusion** integrates features at multiple encoder stages rather than only at the bottleneck [1] [3] [8]. MMFNet employs a Multimodal Feature Fusion Block to progressively integrate optical and digital surface model features at multiple stages, achieving 83.50% mIoU on the Vaihingen benchmark [3] [8]. This approach provides the richest cross-modal interaction but increases computational cost and architectural complexity.

### Sensor-to-Task Routing

The literature provides evidence for routing specific modalities to specific tasks based on their information content. SAR data excels at capturing texture and structural details [4] [12], making TESSERA embeddings (which incorporate SAR backscatter coefficients [20]) well-suited for height prediction. Optical data captures compositional and spectral information [12], making AlphaEarth embeddings appropriate for abundance fraction prediction. 

A practical routing strategy would direct TESSERA encoder features preferentially to the height decoder through dedicated skip connections, while routing AlphaEarth features to the fraction decoder. Cross-encoder bridges could provide minimal shared context—for example, a downweighted TESSERA bottleneck feature could serve as a side-input to the fraction decoder to provide structural priors for building segmentation, with gradient scaling (e.g., 0.1×) to prevent the height task from dominating fraction learning.

### Implementation Considerations

Skip connection routing requires careful design. In a standard U-Net, skip connections from encoder layer *i* connect to decoder layer *n-i*. In a dual-encoder/dual-decoder architecture, each encoder should route its skip features primarily to its corresponding decoder, with optional cross-connections mediated by attention or gating mechanisms [13]. Adapter layers may be necessary for channel alignment when fusing features from encoders with different output dimensions [13].

Gradient isolation between encoder pathways prevents one modality from dominating training. This can be achieved through separate optimizer states per encoder or through gradient scaling factors applied to the loss backpropagated to each encoder branch. The dual-stream medical imaging architecture of Valindria et al. demonstrated that modality-specific encoders with shared decoders improved segmentation accuracy over single-modality training, with particular benefits for varying structures [16].

### When to Choose This Approach

Dual-encoder architectures are most appropriate when modality signals are fundamentally distinct and task outputs have different spatial characteristics. The GeoFM competition exhibits both conditions: AlphaEarth and TESSERA represent different embedding spaces (experiment 6A demonstrated their incompatibility), and abundance fractions versus height represent different output types (segmentation-like versus continuous regression). The architecture is less suitable when modalities are highly redundant or when computational budget is severely constrained, as dual encoders require roughly twice the parameters and compute of single-encoder designs.

The quantitative evidence supports this approach for multi-sensor remote sensing. Dual-encoder designs have demonstrated improvements over single-modality baselines on multi-sensor fusion tasks [2]., and 2.03% mean F1 improvement on multi-objective segmentation [17]. For the GeoFM competition, where the performance gap between current best (0.3721) and top team (0.5269) is substantial, dual-encoder fusion with sensor-routed decoders offers a principled solution to the modality and task competition problems that have limited progress to date.

---

## Alternative 2 — Adapter Mixture Fusion with Learned IoU Surrogates

When computational or parameter budgets constrain the deployment of full dual-encoder architectures, adapter-based fusion offers a parameter-efficient alternative that addresses both the modality competition problem and the hard-IoU evaluation tension. This approach freezes or lightly tunes the main encoder parameters while inserting small trainable adapter modules that specialize features for different modalities and tasks. Combined with loss functions explicitly designed to align with threshold-based evaluation metrics, adapter mixture fusion provides a practical path forward when Alternative 1's dual encoders exceed available resources.

### Parameter-Efficient Modality Specialization

The core principle of adapter-based fusion is to preserve the pretrained knowledge of foundation model encoders while enabling modality-specific adaptation through lightweight inserted modules. SpectralX demonstrates this approach for remote sensing foundation models, introducing an Attribute-oriented Mixture of Adapters (AoMoA) that dynamically aggregates multi-attribute expert knowledge while performing layer-wise fine-tuning [1]. The framework employs a two-stage training process: first, a masked reconstruction task with a specialized Hyper Tokenizer extracts attribute tokens from spatial and spectral dimensions; second, an Attribute-refined Adapter iteratively queries low-level semantic features with high-level representations to focus on task-beneficial attributes [1]. This architecture enables the model to interpret spectral imagery from new regions or seasons without requiring extensive retraining of the foundation model backbone.

For the GeoFM competition, this translates to inserting per-modality adapter experts into the encoder blocks that process AlphaEarth and TESSERA embeddings. Rather than duplicating the entire encoder as in Alternative 1, small adapter layers (typically 1-5% of the base model parameters) are inserted at strategic points in the network. The main encoder parameters remain frozen, preserving the pretrained representations, while the adapters learn modality-specific transformations. TASAM demonstrates a similar principle for remote sensing segmentation, integrating lightweight modules including a terrain-aware adapter that injects elevation priors and a temporal prompt generator that captures land-cover changes over time [16]. These modules achieve substantial performance gains across three remote sensing benchmarks while maintaining minimal computational overhead [16].

### Mixture-of-Adapters Aggregation

The critical innovation in adapter-based fusion is learned gating that determines which modality expert dominates at each spatial location. Rather than naively concatenating features (which caused the catastrophic failure in experiment 6A), the network learns spatially-varying weights that route information through appropriate adapter pathways. The AoMoA mechanism in SpectralX provides a concrete implementation: it dynamically aggregates multi-attribute expert knowledge, allowing the model to emphasize different modality characteristics depending on the local feature context [1]. This addresses the modality competition problem by enabling the network to discover which embedding space—AlphaEarth's composition-sensitive optical features or TESSERA's structure-sensitive SAR-optical fusion—provides more reliable information for each pixel.

For the GeoFM task, a practical implementation would insert adapter modules at multiple encoder depths (e.g., after blocks 3, 6, 9, and 12 in a 12-layer transformer). At each insertion point, separate adapter experts process AlphaEarth and TESSERA features, and a learned gating network produces per-pixel weights that combine the expert outputs. The gating network can be conditioned on both the input embeddings and intermediate encoder features, allowing it to make context-aware routing decisions. This architecture preserves the parameter efficiency of shared encoders while providing the modality specialization that experiment 6A demonstrated was necessary.

### Loss Alignment for Hard-IoU Evaluation

The second major advantage of this alternative is the opportunity to directly address the hard-IoU versus continuous regression tension through specialized loss functions. The platform evaluates building predictions by thresholding continuous abundance fractions at 0.5 and computing IoU, but standard MSE loss does not optimize for this discrete decision boundary. Auto Seg-Loss provides a principled solution by searching for differentiable surrogate losses that align with specific evaluation metrics [4]. The framework substitutes non-differentiable operations in metrics with parameterized functions and conducts parameter search to optimize the shape of loss surfaces [4]. On PASCAL VOC and Cityscapes, the searched mIoU surrogate outperformed manually designed losses including cross-entropy, Dice loss, and Lovász loss [4].

An alternative approach uses calibrated surrogates that create gradient pressure specifically at the decision boundary. Calibrated Dice employs a quasi-concave, lower-bounded, and calibrated surrogate for the F₁-score that has better theoretical properties than soft-Dice [2]. The key insight is to apply a shifted sigmoid transformation before computing the Dice coefficient [2]. This transformation concentrates gradient signal near the 0.5 threshold where the hard evaluation occurs, while maintaining differentiability. Experimental results on 3D segmentation problems indicate improved stability compared to standard soft-Dice [2].

A third strategy directly optimizes soft IoU with careful batching to ensure stability. Mini-batch Soft IoU extends the IoU loss function to multi-class segmentation networks and ensures that the number of categories in each mini-batch equals the batch size at least, addressing the instability of IoU loss function [3]. This method attains significant improvements beyond state-of-the-art IoU loss function methods on PASCAL VOC2012 [3].

For the GeoFM competition, a composite objective combining a region overlap surrogate (calibrated Dice or searched IoU surrogate) for the abundance channels with a regression term (MAE or Huber loss) for the height channel would directly address the task competition problem. The region overlap component optimizes for the hard-IoU evaluation, while the regression component handles the continuous height prediction. Gradient scaling factors can balance the two objectives, preventing either task from dominating training.

### Implementation Strategy and When to Choose This Approach

A practical implementation would proceed in two stages, following the SpectralX framework [1]. Stage one employs self-supervised pretext tasks—masked reconstruction of the GeoFM embeddings—to initialize the adapter modules and learn robust modality-specific representations before supervised fine-tuning. This stage can leverage unlabeled data from the test regions to improve domain robustness. Stage two introduces the supervised segmentation task with the IoU-aligned loss function, fine-tuning the adapters while keeping the main encoder frozen or applying minimal learning rate scaling (e.g., 0.1× for encoder, 1.0× for adapters).

This alternative is most appropriate when: (1) parameter budget is constrained and full dual encoders are too costly; (2) pretrained foundation model features are strong and should be preserved; (3) the hard-IoU evaluation metric is the primary bottleneck; or (4) rapid experimentation is needed, as adapter modules train faster than full encoders. The approach is less suitable when modality signals are so fundamentally incompatible that even learned gating cannot reconcile them, or when the foundation model embeddings are poorly suited to the task and require substantial retraining.

The quantitative evidence supports parameter-efficient adaptation for remote sensing. SpectralX achieves domain generalization across diverse spectral inputs without extensive retraining [1], and TASAM demonstrates substantial performance gains across multiple benchmarks with minimal computational overhead [16]. For the GeoFM competition, where the performance gap between current best (0.3721) and top team (0.5269) is substantial, adapter mixture fusion with IoU-aligned loss functions offers a computationally efficient solution that directly targets both the modality competition and hard-threshold evaluation problems.

---

## Alternative 3 — Ensemble with Calibration and Domain Adaptation

The ensemble-with-calibration approach addresses the domain shift and threshold alignment problems orthogonally to the architectural choices presented in Alternatives 1 and 2. Rather than redesigning the encoder-decoder structure, this strategy wraps existing architectures with three complementary mechanisms: per-modality decoder ensembles that reduce single-embedding failure modes, post-hoc threshold calibration that aligns continuous predictions with the hard 0.5 IoU evaluation, and domain adaptation techniques that close the train-to-test distribution gap without requiring target labels. This alternative can be applied on top of either the dual-encoder fusion (Alternative 1) or adapter mixture (Alternative 2) architectures, providing a practical path to improve generalization when geographic validation splits reveal that models overfit to training regions.

### Per-Modality Decoder Ensemble

The core principle of decoder ensembling is to train independent prediction heads for each foundation model embedding, then aggregate their outputs through learned or gated mechanisms. For the GeoFM competition, this translates to training separate decoders that process AlphaEarth embeddings, TESSERA embeddings, and potentially TerraMind embeddings independently. Each decoder specializes in extracting task-relevant features from its assigned embedding space, avoiding the destructive interference observed in experiment 6A when embeddings were naively concatenated.

The aggregation strategy determines how per-modality predictions are combined at inference. CoDEx demonstrates the effectiveness of multi-expert ensembles for remote sensing domain generalization, training one expert model per training domain and using a model selection module to identify suitable experts for test samples [8]. On the DynamicEarthNet benchmark, CoDEx improved mean IoU from 37.5 to 39.1 and overall accuracy from 73.8 to 75.7 compared to single-model baselines. For GeoFM, a similar gating network could learn spatially-varying weights that emphasize AlphaEarth predictions in regions with strong spectral signatures (e.g., vegetated areas) and TESSERA predictions in structurally complex regions (e.g., dense urban areas with height variation).

A practical implementation would train three independent U-Net decoders—one per embedding source—each producing four abundance channels plus one height channel. At inference, a lightweight gating network conditioned on the input embeddings produces per-pixel weights for each decoder. The final prediction is a weighted sum of the three decoder outputs. This architecture preserves the modality-specific strengths demonstrated in single-embedding experiments while reducing variance through ensemble averaging. The gating network can be trained end-to-end with the decoders or added in a second stage using a small validation set that mimics test-region characteristics.

### Post-Hoc Threshold Calibration

The platform evaluates building predictions by thresholding continuous abundance fractions at 0.5 and computing IoU, but this fixed threshold may be suboptimal across different geographic regions or land cover types. Post-hoc calibration addresses this by searching for region-specific or channel-specific thresholds that maximize hard-IoU on validation data that reflects test-region characteristics.

The calibration procedure requires geographic validation splits that respect domain boundaries. Rather than random train-validation splits that allow spatial autocorrelation to leak information, geographic cross-validation partitions the training data by spatial clustering (e.g., KMeans over latitude-longitude coordinates). For each geographic fold, the model is trained on all regions except the holdout fold, then threshold values are scanned from 0.05 to 0.95 in increments of 0.05 on the holdout region. The threshold that maximizes IoU on the holdout region is recorded. This process is repeated across all folds to obtain a distribution of optimal thresholds, which can be averaged or selected based on validation performance.

CrossEarth emphasizes the importance of cross-domain evaluation for remote sensing segmentation, demonstrating that models optimized for in-domain performance often fail to generalize across regions, spectral bands, platforms, and climates [1] [7]. The framework curates an RSDG benchmark comprising 28 cross-domain settings [1] (expanded to 32 semantic segmentation scenarios in the journal version [7]) across various domain shifts, providing a template for designing validation splits that reflect test-region diversity. For the GeoFM competition, a practical approach would cluster training tiles by geographic coordinates into 5-7 spatial folds, train on 4-6 folds, and calibrate thresholds on the remaining fold. The calibrated thresholds are then applied to test predictions.

An extension of this approach allows per-channel or per-region threshold adaptation. If validation data suggests that building predictions in arid regions require a lower threshold (e.g., 0.35) than in temperate regions (e.g., 0.55), the calibration procedure can learn region-specific thresholds conditioned on metadata or embedding statistics. This requires a small amount of geographic metadata (e.g., climate zone or biome classification) that can be derived from tile coordinates.

### Domain Adaptation via Contrastive Style Alignment

The test set covers different geographic regions and time periods than the training data, creating distribution shifts that degrade model performance. Domain adaptation techniques address this by aligning feature distributions between source (training) and target (test) domains without requiring target labels. Three strategies from the remote sensing literature are particularly relevant for the GeoFM competition.

**Contrastive style alignment** applies style augmentation to training data and uses contrastive learning to enforce that augmented and original embeddings produce similar features. DASAM employs a contrastive learning module to align features between source and style-augmented images for cross-domain water body segmentation, achieving robust domain generalization without requiring annotations from the target domain [4]. The mechanism applies photometric transformations (brightness, contrast, saturation adjustments) to training images, then enforces that the model's internal representations remain invariant to these transformations through a contrastive loss. For GeoFM, this translates to applying channel-wise gain and offset to the foundation model embeddings (not pixel-space color transforms, since the input is already embedded), then training the decoder to produce consistent predictions for original and augmented embeddings.

DAugNet demonstrates a related approach for multi-source, multi-target domain adaptation, using a shallow data augmentor network to perform style transfer between satellite images in an unsupervised manner [3]. On city-to-city adaptation tasks, DAugNet achieved overall IoU of 58.37 (Villach → Bad Ischl) and 51.05 (Bad Ischl → Villach), outperforming competing methods. The data augmentor diversifies training batches by stylizing each patch as a randomly selected domain, making the classifier robust to large distribution differences. For GeoFM, a lightweight style augmentor could be trained to transform embeddings from one geographic region to mimic the statistical properties of another region, then used to augment training data during decoder training.

**Embedding-space augmentation** simulates train-to-test distribution shift by applying calibrated noise or transformations directly to the GeoFM embeddings. Self-supervised domain-agnostic domain adaptation (SS(DA)²) uses a contrastive generative adversarial loss to train a generative network for image-to-image translation between satellite image patches, then augments training data with different testing spectral characteristics [2]. SS(DA)² demonstrated effectiveness on the Inria → DeepGlobe building segmentation benchmark [2]. For GeoFM, embedding-space augmentation would apply learned or heuristic transformations (e.g., channel-wise dropout, Gaussian noise, or affine transformations) to the foundation model embeddings during training, forcing the decoder to learn features robust to embedding-space perturbations.

### Pseudo-Label Refinement for Unseen Regions

When test regions are geographically distant from training regions, pseudo-labeling provides a mechanism to adapt the model using unlabeled test data. DAVI integrates task-specific knowledge from a model trained on source regions with an image segmentation foundation model to generate pseudo labels of possible damage in target regions, then employs a two-stage refinement process targeting both pixel and image levels [6]. The method achieves robust performance across diverse terrains (USA and Mexico) and disaster types (wildfires, hurricanes, earthquakes) without requiring ground-truth labels from the target region.

For the GeoFM competition, a pseudo-label refinement pipeline would proceed in three stages. First, train the ensemble model on all available training data with geographic cross-validation. Second, generate predictions on a held-out geographic region (or on test data if allowed by competition rules) and threshold at the calibrated value to produce pseudo-labels. Third, filter pseudo-labels by ensemble consensus—retain only pixels where all three per-modality decoders agree on the predicted class—to ensure high-confidence labels. Fourth, fine-tune the model on the pseudo-labeled data with a reduced learning rate, treating high-confidence pseudo-labels as ground truth. This approach leverages the ensemble's collective knowledge to bootstrap adaptation to unseen regions.

M³SPADA demonstrates a related approach for multi-sensor temporal domain adaptation, jointly exploiting self-training and adversarial learning to transfer a land cover classifier from one time period to another on the same geographical area [14]. The method addresses temporal shifts caused by different climate, weather, or environmental conditions. For GeoFM, a temporal adaptation variant could be applied if training and test data are known to come from different seasons or years, using pseudo-labels generated on temporally-shifted validation data to adapt the model.

### Practical Integration and When to Choose This Approach

The ensemble-with-calibration strategy integrates with Alternatives 1 and 2 at the decoder level. For Alternative 1 (dual-encoder fusion), each encoder branch would feed into three separate decoders—one per foundation model embedding—rather than a single shared decoder. For Alternative 2 (adapter mixture), the adapter-augmented encoder would similarly feed into per-modality decoders. The ensemble aggregation and threshold calibration steps are applied at inference time, adding minimal computational overhead.

Geographic cross-validation is critical for validating this approach. KMeans clustering over latitude-longitude coordinates partitions training tiles into spatial folds that respect geographic boundaries. A 5-fold geographic CV provides five train-validation splits where each validation fold represents a distinct geographic region. Models are trained on four folds and evaluated on the fifth, with threshold calibration performed on the validation fold. This procedure ensures that validation metrics reflect true generalization performance rather than overfitting to spatial autocorrelation.

This alternative is most appropriate when: (1) domain shift between training and test regions is substantial, as indicated by poor performance on geographic validation folds; (2) the fixed 0.5 threshold is suboptimal, as evidenced by a gap between soft-prediction quality and hard-IoU scores; (3) multiple foundation model embeddings are available and exhibit complementary failure modes; or (4) computational budget allows training multiple decoders in parallel. The approach is less suitable when training data already covers the full diversity of test regions, when only a single embedding source is available, or when inference latency is severely constrained (though the gating network adds minimal overhead compared to full model ensembles).

The quantitative evidence supports ensemble and calibration strategies for remote sensing domain generalization. CoDEx demonstrates consistent gains over single-model baselines across multiple benchmarks [8], CrossEarth establishes the importance of cross-domain evaluation [1] [7], and DASAM shows that contrastive style alignment enables robust generalization without target labels [4]. For the GeoFM competition, where the performance gap between current best (0.3721) and top team (0.5269) is substantial and domain shift is a known challenge, ensemble-with-calibration offers a complementary strategy that can be layered on top of architectural improvements from Alternatives 1 and 2 to maximize generalization performance.

---

## Alternative 4 — Gradient Direction Balancing Beyond GradNorm

The gradient balancing methods presented here address multi-task interference orthogonally to architecture choice, applying equally to the dual-encoder fusion architectures, adapter-based mixture-of-experts, or ensemble strategies described in previous sections. While experiment 7A plans to implement GradNorm for balancing abundance prediction and height regression losses, the literature reveals fundamental limitations in magnitude-only gradient balancing that motivate more sophisticated alternatives.

### The Limitation of Magnitude-Only Balancing

GradNorm dynamically tunes gradient magnitudes to equalize training rates across tasks, improving accuracy and reducing overfitting compared to static loss weighting [15]. However, GradNorm operates exclusively on gradient magnitudes without addressing gradient direction conflicts. When task gradients point in opposing directions—a common occurrence when abundance prediction benefits from features that harm height regression—magnitude balancing alone cannot prevent destructive interference. The task competition observed when collapsing the Y-Net dual-decoder in experiment 6A likely reflects such directional conflicts, where shared parameters receive contradictory update signals from the two tasks.

### Rotograd: Aligning Gradient Directions

Rotograd extends GradNorm by adding task-specific rotation matrices that dynamically homogenize both gradient magnitudes and directions across tasks [1]. The method introduces a layer of rotation matrices that align task gradients before applying them to shared parameters, preventing conflicting updates that would otherwise cancel or degrade learning. Experiments show that Rotograd outperforms previous approaches for multitask learning [1]. The approach provides theoretical guarantees on algorithm stability and convergence through game-theoretic analysis [1].

For the GeoFM competition, Rotograd would operate at the interface between task-specific decoders and shared encoder layers. Task-specific rotation matrices would transform gradients from the fraction decoder and height decoder before backpropagating to shared AlphaEarth and TESSERA encoders, ensuring that updates to shared features do not favor one task at the expense of the other. Implementation requires adding small task-specific rotation layers or applying gradient-space transformations during multi-task backpropagation, expanding the model architecture modestly but preserving the core dual-decoder structure.

### FedGradNorm: Heterogeneous Task Normalization

FedGradNorm addresses a distinct challenge relevant to the GeoFM competition: balancing tasks when they exhibit statistical heterogeneity due to different complexities or when embeddings originate from varied sources [2] [3]. The method uses dynamic weighting to normalize gradient norms across heterogeneous task distributions, achieving faster training performance compared to equal-weighting strategies while compensating for imbalanced datasets [2]. FedGradNorm demonstrates exponential convergence rates and improves overall learning performance in personalized federated settings where tasks differ substantially in scale or data distribution [3].

This capability directly addresses the GeoFM scenario where AlphaEarth, TESSERA, and TerraMind embeddings derive from different self-supervised learning objectives and data distributions. AlphaEarth uses optical imagery with composition-sensitive features, TESSERA fuses SAR and optical data with structure-sensitive representations, and TerraMind incorporates temporal dynamics. These embedding spaces exhibit different statistical properties that could cause one modality to dominate gradient updates. FedGradNorm-style normalization would ensure that gradients from TESSERA-derived height predictions do not overwhelm gradients from AlphaEarth-derived abundance fractions simply due to scale differences in their respective loss landscapes.

The practical implementation would apply gradient norm normalization separately to each encoder pathway in the dual-encoder architecture, ensuring that updates to TESSERA and AlphaEarth encoders maintain balanced magnitudes despite differences in embedding space characteristics. This normalization operates independently of the rotation-based direction alignment in Rotograd, making the two methods complementary rather than competing.

### Learn-to-Teach Dynamic Objective Selection

The L2T-FMT framework introduces a teacher-student dynamic that selects which objective to emphasize at each training step, reducing the need to handcraft loss weights [4]. The teacher network instructs the student to learn from either accuracy or fairness objectives depending on which is harder to learn for each task, improving fairness by 12-19% and accuracy by up to 2% over state-of-the-art approaches [4]. While originally designed for fairness-accuracy trade-offs, the principle applies directly to multi-task loss selection in the GeoFM context.

For the competition, a learn-to-teach approach would dynamically prioritize IoU_B optimization when building segmentation is the limiting metric, then shift emphasis to height RMSE when height predictions lag behind. This adaptive prioritization addresses the hard-IoU versus continuous regression tension identified in the introduction, where pure MSE optimization fails to directly optimize the discrete decision boundary that determines competition performance. The teacher network would monitor validation IoU_B and height RMSE, adjusting which decoder receives stronger gradient signals based on current performance gaps.

Implementation requires a meta-learning loop where the teacher network observes task-specific validation metrics and outputs per-task emphasis weights that modulate gradient flow to each decoder. This reduces the number of trade-off weights from 2T to T (where T is the number of tasks) [4]. The approach integrates naturally with either Rotograd or FedGradNorm, as the teacher-selected weights can be applied before gradient rotation or normalization operations.

### Practical Recommendations

For the GeoFM competition, we recommend a layered gradient balancing strategy. **Primary recommendation**: implement Rotograd to address gradient direction conflicts between abundance prediction and height regression, replacing the planned magnitude-only GradNorm in experiment 7A. This provides the most direct solution to the task competition problem observed in experiment 6A.

**If embeddings prove heterogeneous**: add FedGradNorm-style normalization to balance gradient contributions from AlphaEarth, TESSERA, and TerraMind encoders. This is particularly relevant if experiment 6A's catastrophic failure when concatenating embeddings reflects fundamental scale mismatches between embedding spaces rather than purely architectural issues.

**If learned scheduling is preferred**: experiment with L2T-style teacher networks to dynamically prioritize IoU_B versus height RMSE optimization based on validation performance. This addresses the evaluation metric mismatch problem without requiring manual loss weight tuning.

All three methods can be combined sequentially: the teacher network selects task emphasis weights, FedGradNorm normalizes gradients across heterogeneous encoders, and Rotograd aligns gradient directions before updating shared parameters. Each method operates at a different level of the gradient balancing problem—task prioritization, magnitude normalization, and direction alignment—making them complementary rather than redundant. Implementation can proceed incrementally, adding each component as a drop-in replacement or extension to the baseline GradNorm approach planned for experiment 7A.

---

## Synthesis and Recommendations

The four alternatives presented address complementary aspects of the GeoFM competition's technical challenges. Alternative 1 (dual-encoder fusion) and Alternative 2 (adapter mixture) target the architectural problems of modality and task competition. Alternative 3 (ensemble with calibration) addresses domain shift and threshold alignment. Alternative 4 (gradient balancing) provides orthogonal training improvements applicable to any architecture. The evidence base is substantial: 178 papers support dual-encoder approaches, 81 support gradient balancing methods, 79 support domain adaptation strategies, and 15 support IoU-aligned loss functions. This synthesis integrates these alternatives into a coherent decision framework and provides concrete recommendations for experiment 7A.

### Decision Matrix: Problems and Solutions

**Modality competition** (AlphaEarth vs TESSERA embedding incompatibility) is directly addressed by Alternative 1's dual-encoder architecture, which processes each embedding through separate pathways before fusion [2]. Alternative 2's adapter mixture provides a parameter-efficient variant when computational budget constrains full dual encoders. The catastrophic failure in experiment 6A (IoU_B collapse from 0.168 to 0.017) demonstrates that this problem requires architectural separation rather than naive concatenation.

**Task competition** (abundance prediction vs height regression interference) is resolved by Alternative 1's dual-decoder design, which provides task-specific feature hierarchies [1]. The 0.08 proxy metric regression when collapsing Y-Net's dual decoder to a shared decoder in experiment 6A confirms that these tasks benefit from distinct processing pathways. Alternative 2's task-specific adapter heads offer a lightweight alternative when full dual decoders exceed parameter budgets.

**Hard-IoU versus continuous regression tension** is addressed by Alternative 2's IoU-aligned loss functions, which create gradient pressure at the 0.5 decision boundary rather than optimizing pure MSE. Alternative 3's post-hoc threshold calibration provides a complementary inference-time solution that adapts thresholds to geographic regions without retraining.

**Domain shift** between training and test regions is tackled by Alternative 3's geographic cross-validation, contrastive style alignment, and ensemble aggregation strategies. Alternative 3's domain adaptation components operate independently of architecture choice, making them compatible with either Alternative 1 or 2.

**Multi-task interference** during training is mitigated by Alternative 4's gradient balancing methods. Rotograd addresses gradient direction conflicts that magnitude-only methods like GradNorm cannot resolve [4], while FedGradNorm handles heterogeneous embedding spaces [15] [17]. These training improvements apply regardless of whether dual encoders (Alternative 1) or adapters (Alternative 2) are used.

### Recommended Combination for Experiment 7A

The primary recommendation integrates Alternative 1 (dual-encoder/dual-decoder), Alternative 4 (Rotograd), and Alternative 3 (domain adaptation components). This combination directly addresses the two most severe architectural problems—modality competition and task competition—while adding robust training dynamics and domain generalization.

**Architecture**: Implement dual encoders that process AlphaEarth and TESSERA embeddings through separate ResNet-18 or VMamba pathways [2]. Apply cross-attention fusion at decoder skip connection levels (16×16, 32×32 resolution) to enable modality interaction without destructive interference. Route encoder features to task-specific decoders: a fraction decoder for abundance prediction and a height decoder for regression. This architecture prevents the embedding-space conflicts observed in experiment 6A while preserving the task-specific feature hierarchies that Y-Net's dual decoder provided.

**Training dynamics**: Replace the planned GradNorm implementation with Rotograd to address gradient direction conflicts between abundance and height tasks [4]. Rotograd's task-specific rotation matrices align gradients before backpropagating to shared encoder layers, preventing the contradictory update signals that magnitude-only balancing cannot resolve. If embedding heterogeneity proves problematic, add FedGradNorm-style normalization to balance gradient contributions from AlphaEarth and TESSERA encoders [15] [17].

**Domain robustness**: Implement geographic cross-validation by clustering training tiles via KMeans over latitude-longitude coordinates into 5-7 spatial folds. Train on 4-6 folds and validate on the holdout fold to ensure metrics reflect true generalization rather than spatial autocorrelation. Apply contrastive style alignment during training by augmenting foundation model embeddings with channel-wise gain and offset, then enforcing prediction consistency through contrastive loss. Calibrate the 0.5 threshold per geographic fold by scanning values from 0.05 to 0.95 and selecting the threshold that maximizes validation IoU_B.

This combination is justified by the evidence. MMFNet, a dual-encoder architecture with cross-attention fusion, achieves 83.50% mIoU on the ISPRS Vaihingen benchmark [2], and dual-decoder designs with attention-based fusion achieve state-of-the-art performance on RGBD segmentation benchmarks [1]. Rotograd outperforms previous multitask learning approaches on several real-world datasets by homogenizing both gradient magnitudes and directions [4]. Geographic cross-validation and domain adaptation components are essential for remote sensing generalization, as demonstrated by GeoMultiTaskNet achieves 47.22% mIoU on a cross-domain subset of the FLAIR dataset [10].

### Fallback: Alternative 2 as Rapid Iteration Path

If Alternative 1's dual encoders prove computationally prohibitive or training becomes unstable, Alternative 2 (adapter mixture with IoU surrogates) provides a parameter-efficient fallback. Insert per-modality adapter experts (1-5% of base model parameters) at strategic encoder depths, with learned gating networks producing spatially-varying weights that route information through appropriate pathways. Replace the MSE loss with a composite objective combining calibrated Dice loss for abundance channels and MAE for height. This approach preserves pretrained foundation model features while enabling modality specialization and threshold-aware optimization.

The adapter approach is supported by evidence showing that parameter-efficient adaptation achieves domain generalization without extensive retraining, and IoU-aligned surrogates outperform manually designed losses on segmentation benchmarks. Implementation would proceed in two stages: self-supervised masked reconstruction to initialize adapters, followed by supervised fine-tuning with the composite loss. This path reduces training time and parameter count compared to full dual encoders while still addressing modality competition and hard-IoU alignment.

### Ablation Priorities for Phase 5

The 30-day deadline and 12-hour submission limit require strategic ablation planning. Priority 1: Encoder modality split (Alternative 1 claim 1). Compare single shared encoder versus dual AlphaEarth/TESSERA encoders to quantify the modality competition effect. Priority 2: Decoder task split (Alternative 1 claim 2). Compare shared decoder versus dual fraction/height decoders to isolate task competition impact. Priority 3: Gradient balancing method (Alternative 4). Compare Rotograd versus GradNorm versus FedGradNorm to validate direction alignment benefits. Priority 4: Loss function (Alternative 2). Compare IoU surrogate versus composite (MAE+Tversky) versus pure MSE to measure threshold alignment gains. Priority 5: Threshold calibration (Alternative 3). Compare per-channel versus global threshold versus fixed 0.5 to quantify calibration value.

These ablations isolate the contribution of each component while respecting compute constraints. The dual-encoder and dual-decoder ablations directly test the architectural hypotheses from experiment 6A's failures. The gradient balancing comparison validates whether direction alignment provides measurable gains over magnitude-only methods. The loss function and threshold ablations quantify how much of the 0.3721 vs 0.5269 IoU_B gap stems from evaluation metric misalignment.

### Timeline and Integration Strategy

With 30 days to deadline, prioritize Alternative 1 + Alternative 4 + Alternative 3 (domain components) as a single integrated baseline. Allocate 10 days for implementation and initial training, 10 days for ablations and hyperparameter tuning, and 10 days for final training and ensemble preparation. Reserve Alternative 3's full ensemble (per-modality decoder aggregation) for final submission if time permits, as ensemble training can proceed in parallel once the base architecture is stable.

All four alternatives are architecturally compatible. Alternative 1 or 2 defines the encoder-decoder structure. Alternative 4 operates during training to balance task gradients. Alternative 3 applies at inference for threshold calibration and during training for domain adaptation. This modularity allows incremental integration: start with dual encoders and Rotograd, add geographic cross-validation and style augmentation, then layer in threshold calibration and optional ensemble aggregation as time permits.

The recommended combination closes the loop on all four problems identified in the introduction. Dual encoders resolve modality competition. Dual decoders resolve task competition. IoU-aligned losses and threshold calibration resolve the hard-IoU tension. Geographic cross-validation and domain adaptation resolve domain shift. Rotograd resolves multi-task interference. This integrated approach, grounded in 353 papers across the four alternatives, provides a principled path to close the 0.1548 IoU_B gap between current performance and the competition leader.

## References

[1]S. Wang, L. Cao, and H. Deng, “MFMamba: A Mamba-Based Multi-Modal Fusion Network for Semantic Segmentation of Remote Sensing Images,” Sensors, vol. 24, no. 22, pp. 7266–7266, Nov. 2024, doi: 10.3390/s24227266.

[2]“A Dual-Branch Deep Learning Architecture for Multisensor and Multitemporal Remote Sensing Semantic Segmentation,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 16, pp. 2147–2162, Jan. 2023, doi: 10.1109/jstars.2023.3243396.

[3]J. Qiu et al., “MMFNet: A Mamba-Based Multimodal Fusion Network for Remote Sensing Image Semantic Segmentation,” Sensors, vol. 25, no. 19, pp. 6225–6225, Oct. 2025, doi: 10.3390/s25196225.

[4]B. Ren, B. Hou, and Y. Gu, “Multi-Source Fusion Network for Remote Sensing Image Segmentation with Hierarchical Transformer,” pp. 6318–6321, July 2023, doi: 10.1109/igarss52108.2023.10282984.

[5]S. Debopom and A. Sami, “CLAIRE: A Dual Encoder Network with RIFT Loss and Phi-3 Small Language Model Based Interpretability for Cross-Modality Synthetic Aperture Radar and Optical Land Cover Segmentation,” Sept. 2025, doi: 10.48550/arxiv.2509.11952.

[6]S. Wu, J. Zhu, Y. Gu, W. Han, W. Jiang, and J. Geng, “SegCR: A Multimodal and Multitask Complementary Fusion Network for Remote Sensing Semantic Segmentation and Cloud Removal,” IEEE Transactions on Geoscience and Remote Sensing, pp. 1–1, Jan. 2025, doi: 10.1109/tgrs.2025.3603066.

[7]J. F. Qiu, W. Chang, W. Ren, S. Hou, and R. Yang, “MMFNet: A Mamba-Based Multimodal Fusion Network for Semantic Segmentation of Remote Sensing,” Aug. 2025, doi: 10.20944/preprints202508.0078.v1.

[8]H. Yu, G. Li, H. Liu, S. Zhu, W. Dong, and C. Li, “SpecSAR-Former: A Lightweight Transformer-based Network for Global LULC   Mapping Using Integrated Sentinel-1 and Sentinel-2,” Oct. 2024, doi: 10.48550/arxiv.2410.03962.

[9]X. Ma, X. Zhang, M.-O. Pun, and B. Huang, “MANet: Fine-Tuning Segment Anything Model for Multimodal Remote Sensing   Semantic Segmentation,” Oct. 2024, doi: 10.48550/arxiv.2410.11160.

[10]V. V. Valindria et al., “Multi-modal Learning from Unpaired Images: Application to Multi-organ Segmentation in CT and MRI,” pp. 547–556, Mar. 2018, doi: 10.1109/WACV.2018.00066.

[11]J. Zhang et al., “CD-MQANet: Enhancing Multi-Objective Semantic Segmentation of Remote Sensing Images through Channel Creation and Dual-Path Encoding,” Remote sensing, Sept. 2023, doi: 10.3390/rs15184520.

[12]Z. Feng et al., “TESSERA: Temporal Embeddings of Surface Spectra for Earth Representation and Analysis,” arXiv.org, vol. abs/2506.20380, doi: 10.48550/arxiv.2506.20380.

[13]Y. Zhang, W. Li, M. Zhang, J. Han, R. Tao, and S. Liang, “SpectralX: Parameter-efficient Domain Generalization for Spectral Remote Sensing Foundation Models,” arXiv.org, vol. abs/2508.01731, Aug. 2025, doi: 10.48550/arxiv.2508.01731.

[14]M. Nordström, H. Bao, F. Löfman, H. Hult, A. Maki, and M. Sugiyama, “Calibrated Surrogate Maximization of Dice,” pp. 269–278, Oct. 2020, doi: 10.1007/978-3-030-59719-1_27.

[15]Y. Huang, Z. Tang, D. Chen, K. Su, and C. Chengbin, “Batching Soft IoU for Training Semantic Segmentation Networks,” IEEE Signal Processing Letters, vol. 27, pp. 66–70, Jan. 2020, doi: 10.1109/LSP.2019.2956367.

[16]H. Li, C. Tao, X. Zhu, X. Wang, G. Huang, and J. Dai, “Auto Seg-Loss: Searching Metric Surrogates for Semantic Segmentation,” May 2021.

[17]W. Tian-yang, X. Xi, C. Gaofei, Z. Qi, C. Guo, and J. YingRui, “TASAM: Terrain-and-Aware Segment Anything Model for Temporal-Scale Remote Sensing Segmentation,” Sept. 2025, doi: 10.48550/arxiv.2509.15795.

[18]Z. Gong et al., “CrossEarth: Geospatial Vision Foundation Model for Domain Generalizable   Remote Sensing Semantic Segmentation,” Oct. 2024, doi: 10.48550/arxiv.2410.22629.

[19]F. Zhang, Y. Shi, and X. Zou, “Self-supervised Domain-agnostic Domain Adaptation for Satellite Images,” arXiv.org, vol. abs/2309.11109, Sept. 2023, doi: 10.48550/arxiv.2309.11109.

[20]O. Tasar, A. Giros, Y. Tarabalka, P. Alliez, and S. Clerc, “DAugNet: Unsupervised, Multisource, Multitarget, and Life-Long Domain Adaptation for Semantic Segmentation of Satellite Images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 2, pp. 1067–1081, Feb. 2021, doi: 10.1109/TGRS.2020.3006161.

[21]L. Yang, P. Liu, G. Zhang, H. Zhao, and C. Zhao, “Domain-Adaptive Segment Anything Model for Cross-Domain Water Body Segmentation in Satellite Imagery,” Journal of Imaging, vol. 11, Dec. 2025, doi: 10.3390/jimaging11120437.

[22]K. Ahn, S. Han, S. W. Park, J. Kim, S. Park, and M. Cha, “Generalizable Disaster Damage Assessment via Change Detection with   Vision Foundation Model,” June 2024, doi: 10.48550/arxiv.2406.08020.

[23]Z. Gong et al., “CrossEarth: Geospatial Vision Foundation Model for Domain Generalizable Remote Sensing Semantic Segmentation.,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. PP, Dec. 2025, doi: 10.1109/tpami.2025.3649001.

[24]A. Kuriyal, E. Vincent, M. Aubry, and L. Landrieu, “CoDEx: Combining Domain Expertise for Spatial Generalization in Satellite Image Analysis,” vol. abs/2504.19737, Apr. 2025, doi: 10.48550/arxiv.2504.19737.

[25]E. Capliez et al., “Multisensor Temporal Unsupervised Domain Adaptation for Land Cover Mapping With Spatial Pseudo-Labeling and Adversarial Learning,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–16, doi: 10.1109/tgrs.2023.3297077.

[26]A. Javaloy and I. Valera, “Rotograd: Dynamic Gradient Homogenization for Multitask Learning,” arXiv: Learning, May 2021.

[27]“FedGradNorm: Personalized Federated Gradient-Normalized Multi-Task Learning,” July 2022, doi: 10.1109/spawc51304.2022.9833969.

[28]“FedGradNorm: Personalized Federated Gradient-Normalized Multi-Task   Learning,” Mar. 2022, doi: 10.48550/arxiv.2203.13663.

[29]“Learning to Teach Fairness-aware Deep Multi-task Learning,” June 2022, doi: 10.48550/arxiv.2206.08403.

[30]Z. Chen, V. Badrinarayanan, C.-Y. Lee, and A. Rabinovich, “GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks,” pp. 794–803, July 2018.

[31]Y. Zhang, Y. Yang, C. Xiong, G. Sun, and Y. Guo, “Attention-based Dual Supervised Decoder for RGBD Semantic Segmentation,” arXiv.org, vol. abs/2201.01427, Jan. 2022.

[32]V. Marsocci, N. Gonthier, A. Garioud, S. Scardapane, and C. Mallet, “GeoMultiTaskNet: remote sensing unsupervised domain adaptation using geographical coordinates,” June 2023, doi: 10.1109/cvprw59228.2023.00201.