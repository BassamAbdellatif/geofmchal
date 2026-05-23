# Attention-Gated Skip Connections: Architecture & Training Epochs

This document explains what happens during a training epoch of our multi-modal fusion model and details the layout of the network architecture.

---

## 1. What Happens During a Training Epoch?

An epoch is one complete pass of the training algorithm through the entire training dataset. For a single epoch, the step-by-step process is as follows:

```mermaid
flowchart TD
    A[Start Epoch] --> B[Shuffle Dataset & Create Batches]
    B --> C[Fetch Batch: Pixel & Patch Embeddings + Targets]
    C --> D[Forward Pass: Generate Predictions]
    D --> E[Compute Loss: MAE + SSIM + Gradient + Tversky]
    E --> F[Backward Pass: Calculate Gradients]
    F --> G[Gradient Clipping: max_norm=1.0]
    G --> H[Update Weights via AdamW Optimizer]
    H --> I{More Batches?}
    I -- Yes --> C
    I -- No --> J[Run Validation Split Evaluation]
    J --> K[Log Epoch Time & Metrics to File]
    K --> L[End Epoch]
```

### Detailed Steps:
1. **Data Shuffling & Batching**: The dataloader shuffles the dataset pairs/triplets to ensure the model does not learn the order of the data. It divides the data into batches (default batch size: `16`).
2. **Forward Pass (Inference)**: 
   * The inputs are loaded: pixel embeddings ($256 \times 256 \times C_{pixel}$) and patch embeddings ($16 \times 16 \times C_{patch}$).
   * The network processes the inputs to generate the 4-channel target prediction ($256 \times 256 \times 4$).
3. **Loss Calculation**: The `ImprovedCompositeLoss` function evaluates the prediction against the ground truth labels:
   * **MAE Loss**: Calculates pixel-wise absolute error for all classes.
   * **SSIM Loss**: Captures structural similarities (edges, contours) for spatial accuracy.
   * **Gradient Loss**: Penalizes discrepancies in image gradients (making edges sharper).
   * **Tversky Loss**: Handles class imbalance (highly effective for segmentation of rare classes).
4. **Backward Pass (Backpropagation)**: PyTorch computes the gradients of the loss with respect to all trainable parameters of the model.
5. **Gradient Clipping**: Restricts the maximum norm of the gradients to `1.0` to prevent gradient explosion and stabilize training.
6. **Optimizer Update**: The `AdamW` optimizer uses the calculated gradients and weight decay to update the network weights.
7. **Validation & Scheduler Step**: After running through all training batches, the model runs a validation pass on unseen validation samples, computes the validation loss, and updates the learning rate scheduler (`ReduceLROnPlateau`).

---

## 2. Schematic Network Drawing

Our architecture uses **Option 2A: Attention-Gated Skip Connections**. High-resolution pixel-level features are fused with patch-level semantic features at the bottleneck, and the decoding phase is attention-guided by the high-resolution encoder features.

```mermaid
graph TD
    %% Inputs
    subgraph Inputs
        PixelIn["Pixel Inputs (Tessera + Alpha Earth)<br>Size: 256x256x192"]
        PatchIn["Patch Inputs (TerraMind + Thor)<br>Size: 16x16x3072"]
    end

    %% Encoder
    subgraph Encoder [U-Net Encoder]
        Enc1["Encoder Layer 1<br>256x256x64"]
        Enc2["Encoder Layer 2<br>128x128x128"]
        Enc3["Encoder Layer 3<br>64x64x256"]
        Enc4["Encoder Layer 4<br>32x32x512"]
    end

    %% Bottleneck
    subgraph Bottleneck [Bottleneck & Patch Injection]
        ProjPatch["Linear Projection<br>16x16x512"]
        ConvBot["Encoder Bottleneck<br>16x16x512"]
        ConcatBot["Concatenate + Conv<br>16x16x1024"]
    end

    %% Decoder & Attention Gates
    subgraph Decoder [U-Net Decoder]
        AG4["Attention Gate 4"]
        Dec4["Decoder Layer 4<br>32x32x512"]
        
        AG3["Attention Gate 3"]
        Dec3["Decoder Layer 3<br>64x64x256"]
        
        AG2["Attention Gate 2"]
        Dec2["Decoder Layer 2<br>128x128x128"]
        
        AG1["Attention Gate 1"]
        Dec1["Decoder Layer 1<br>256x256x64"]
    end

    %% Final Output
    FinalConv["1x1 Convolution"]
    Outputs["Predictions<br>256x256x4"]

    %% Flow lines
    PixelIn --> Enc1
    Enc1 -->|Downsample| Enc2
    Enc2 -->|Downsample| Enc3
    Enc3 -->|Downsample| Enc4
    Enc4 -->|Downsample| ConvBot

    PatchIn --> ProjPatch
    ConvBot & ProjPatch --> ConcatBot

    %% Decoder flow
    ConcatBot -->|Upsample| Dec4
    Dec4 -->|Upsample| Dec3
    Dec3 -->|Upsample| Dec2
    Dec2 -->|Upsample| Dec1
    Dec1 --> FinalConv
    FinalConv --> Outputs

    %% Attention Skip Connections
    Enc4 -->|Skip Connection| AG4
    Dec4 -->|Gating Signal| AG4
    AG4 -->|Gated Skip Connection| Dec4

    Enc3 -->|Skip Connection| AG3
    Dec3 -->|Gating Signal| AG3
    AG3 -->|Gated Skip Connection| Dec3

    Enc2 -->|Skip Connection| AG2
    Dec2 -->|Gating Signal| AG2
    AG2 -->|Gated Skip Connection| Dec2

    Enc1 -->|Skip Connection| AG1
    Dec1 -->|Gating Signal| AG1
    AG1 -->|Gated Skip Connection| Dec1

    classDef inputStyle fill:#e6f2ff,stroke:#0066cc,stroke-width:2px,color:#003366;
    classDef encStyle fill:#ffe6e6,stroke:#cc0000,stroke-width:2px,color:#660000;
    classDef botStyle fill:#ffffee,stroke:#b3b300,stroke-width:2px,color:#4d4d00;
    classDef decStyle fill:#e6ffe6,stroke:#009900,stroke-width:2px,color:#004d00;
    
    class PixelIn,PatchIn inputStyle;
    class Enc1,Enc2,Enc3,Enc4 encStyle;
    class ProjPatch,ConvBot,ConcatBot botStyle;
    class AG1,AG2,AG3,AG4,Dec1,Dec2,Dec3,Dec4 decStyle;
```

### Key Components of the Attention Gate (AG)
Each Attention Gate filters the skip connections from the Encoder:
1. **Inputs**: The high-resolution feature map from the Encoder ($x$) and the gating signal from the coarser Decoder layer ($g$).
2. **Operations**:
   * Compute linear transformations of $x$ and $g$ using $1 \times 1$ convolutions.
   * Add the transformed outputs together and pass through a **ReLU** activation.
   * Apply a $1 \times 1$ convolution followed by a **Sigmoid** activation to calculate the attention coefficients $\alpha \in [0, 1]$.
   * Multiply the original encoder feature map $x$ by $\alpha$.
3. **Outcome**: The decoder only receives high-resolution spatial details in regions where the attention gate is active (e.g. focused on building boundaries), reducing noise and improving spatial delineation.
