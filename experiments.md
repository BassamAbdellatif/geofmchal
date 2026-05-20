# Emb2Heights Architecture Experiment Tree

**Goal:** Optimize multi-modal fusion of high-res spatial (Alpha/Tessera) and low-res semantic (TerraMind/THOR) embeddings.

## The Decision Tree

* **[ ] Branch 1: Early Fusion** (Upsample 16x16 -> 256x256, Concat at input)
    * *Status:* Skipped (Inefficient, potential semantic dilution).
* **[ ] Branch 3: Cross-Attention Fusion** (Query=Patch, Key/Value=Pixel)
    * *Status:* Pending (Fallback if Branch 2 plateaus).
* **[x] Branch 2: Two-Stream Bottleneck Injection** (Extract spatial, inject patch at bottleneck)
    * *Status:* **ACTIVE BRANCH**
    * **[x] Option 2A: Attention-Gated Skip Connections**
        * *Status:* **CURRENTLY CODING** * *Hypothesis:* Filtering spatial skip connections using S1 context will preserve height gradients.
    * **[ ] Option 2B: Multi-Scale Feature Injection (FPN)**
        * *Status:* Pending (Use if Option 2A struggles to resolve large building footprints).
    * **[ ] Option 2C: Deep Supervision**
        * *Status:* Pending (Use if validation loss stalls early in training).

---

## Run Log (Leaderboard Tracking)
| Run ID / Git Branch | Architecture Node | Val MAE | Val Tversky | Leaderboard Score | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline_v1` | N/A (Baseline) | [TBD] | [TBD] | [TBD] | 1 epoch local run, basic residual decoder. |
| `exp_2A_attngate` | Node 2A | - | - | - | Initializing... |