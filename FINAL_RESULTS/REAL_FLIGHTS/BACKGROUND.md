# Algorithm Comparison for AUROC Calculation on AOT

We evaluate AUROC using two definitions:

- **Definition 1 (Correct):**  
  - ID Classes: `0,1`  
  - OOD Class: `2,3 


- We evaluate **unmodified RT-DETR** and **RT-DETR with Spectral Normalization (SN)**
- For each model, we compute AUROC:
  - Without and with **temperature scaling**
  - Without and with **pruning detections with Softmax < 20%**
- When pruning improves results, we also report the **new mAP**

---

## 🔹 Definition 1 – Correct Class Mapping

### ➤ No Mod (Original RT-DETR) Baseline mAP: 0.542

#### RT-DETR Logits (mAP in parenthesis)
| Method              | AUROC (Pruned <20) | Temp-Scaled AUROC (Pruned <20) |
|---------------------|----------------------|----------------------------------|
| Softmax             |   0.849              |   0.851                          |
| Logsumexp (density) |   0.846              |   0.852                          |
| Entropy (Energy)    |   0.835              |   0.836                          |




### GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC (Pruned <20) | Entropy AUROC (Pruned <20) |
|-----------------------|-------------------------------|-----------------------------|
| Layer 1               |     0.812                     |  0.591                      |
| Layer 2               |     0.826                     |  0.621                      |
| Layer 3               |     0.819                     |  0.591                      |
| Layer 4               |     0.822                     |  0.541                      |
| Layer 5               |     0.802                     |  0.530                      |
| Layer 6               |     0.711                     |  0.544                      |
| Layer 1 (Temp-Scaled) |     0.810                     |  0.776                      |
| Layer 2 (Temp-Scaled) |     0.823                     |  0.804                      |
| Layer 3 (Temp-Scaled) |     0.816                     |  0.830                      |
| Layer 4 (Temp-Scaled) |     0.822                     |  0.853                      |
| Layer 5 (Temp-Scaled) |     0.802                     |  0.836                      |
| Layer 6 (Temp-Scaled) |     0.709                     |  0.853                      |





> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above) (mAP in parenthesis)
| Num Gaussians Per Class | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|----------------------------------|
| 2                       |   0.851                          |
| 3                       |   0.844                          |
| 4                       |   0.827                          |




### Softmax + GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC (Pruned <20) | Entropy AUROC (Pruned <20) |
|-----------------------|-------------------------------|-----------------------------|
| Layer 1               |   0.879                       |  0.871                      |
| Layer 2               |   0.856                       |  0.885                      |
| Layer 3               |   0.853                       |  0.883                      |
| Layer 4               |   0.857                       |  0.879                      |
| Layer 5               |   0.854                       |  0.866                      |
| Layer 6               |   0.857                       |  0.867                      |
 




---


### ➤ SN Mod (RT-DETR with Spectral Norm) Baseline mAP: 0.520





#### RT-DETR Logits (mAP in parenthesis)
| Method              | AUROC (Pruned <20) | Temp-Scaled AUROC (Pruned <20) |
|---------------------|----------------------|----------------------------------|
| Softmax             |   0.882              |   0.884                          |
| Logsumexp (density) |   0.811              |   0.817                          |
| Entropy (Energy)    |   0.795              |   0.794                          |





### GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC (Pruned <20) | Entropy AUROC (Pruned <20) |
|-----------------------|-------------------------------|-----------------------------|
| Layer 1               |   0.829                       |   0.567                     |
| Layer 2               |   0.805                       |   0.555                     |
| Layer 3               |   0.784                       |   0.531                     |
| Layer 4               |   0.772                       |   0.503                     |
| Layer 5               |   0.780                       |   0.523                     |
| Layer 6               |   0.738                       |   0.538                     |
| Layer 1 (Temp-Scaled) |   0.830                       |   0.792                     |
| Layer 2 (Temp-Scaled) |   0.805                       |   0.757                     |
| Layer 3 (Temp-Scaled) |   0.784                       |   0.790                     |
| Layer 4 (Temp-Scaled) |   0.772                       |   0.796                     |
| Layer 5 (Temp-Scaled) |   0.780                       |   0.844                     |
| Layer 6 (Temp-Scaled) |   0.737                       |   0.821                     |





> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above) (mAP in parenthesis)
| Num Gaussians Per Class | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|----------------------------------|
| 2                       |    0.830                         |
| 3                       |    0.807                         |
| 4                       |    0.818                         |




### Softmax + GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC (Pruned <20) | Entropy AUROC (Pruned <20) |
|-----------------------|-------------------------------|-----------------------------|
| Layer 1               |    0.889                      |  0.887                      |
| Layer 2               |    0.878                      |  0.880                      |
| Layer 3               |    0.886                      |  0.876                      |
| Layer 4               |    0.838                      |  0.867                      |
| Layer 5               |    0.840                      |  0.887                      |
| Layer 6               |    0.870                      |  0.884                      |
 

