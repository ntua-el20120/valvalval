# Algorithm Comparison for AUROC Calculation on AOT

We evaluate AUROC using two definitions:

- **Definition 1 (Correct):**  
  - ID Classes: `0,1`  
  - OOD Class: `2`  
  - Ignored Class: `3`


- We evaluate **unmodified RT-DETR** and **RT-DETR with Spectral Normalization (SN)**
- For each model, we compute AUROC:
  - Without and with **temperature scaling**
  - Without and with **pruning detections with Softmax < 20%**
- When pruning improves results, we also report the **new mAP**

---

## 🔹 Definition 1 – Correct Class Mapping

### ➤ No Mod (Original RT-DETR) Baseline mAP: 0.542

#### RT-DETR Logits (mAP in parenthesis)
| Method                  | AUROC           | Temp-Scaled AUROC | AUROC (Pruned <20) | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|------------------|--------------------|----------------------|-------------------------------|
| Softmax                 |                  |                    | 0.871 (0.519)        |   0.916                       |
| Logsumexp (density)     |                  |                    | 0.814                |   0.870                       |
| Entropy (Energy)        |                  |                    | 0.910                |   0.939                       |



### GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               |                   |                |  0.826                        |   0.751*                    |
| Layer 2               |                   |                |  0.835                        |   0.788*                    |
| Layer 3               |                   |                |  0.822                        |   0.659*                    |
| Layer 4               |                   |                |  0.833                        |   0.616*                    |
| Layer 5               |                   |                |  0.845                        |   0.578*                    |
| Layer 6               |                   |                |  0.840                        |   0.564*                    |
| Layer 1 (Temp-Scaled) |                   |                |  0.815                        |   0.952                     |
| Layer 2 (Temp-Scaled) |                   |                |  0.827                        |   0.925                     |
| Layer 3 (Temp-Scaled) |                   |                |  0.817                        |   0.937                     |
| Layer 4 (Temp-Scaled) |                   |                |  0.830                        |   0.920                     |
| Layer 5 (Temp-Scaled) |                   |                |  0.841                        |   0.927                     |
| Layer 6 (Temp-Scaled) |                   |                |  0.836                        |   0.924                     |




> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above) (mAP in parenthesis)
| Num Gaussians Per Class | Temp-Scaled AUROC | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|--------------------|----------------------------------|
| 2                       |                    |  0.936                           |
| 3                       |                    |  0.857                           |
| 4                       |                    |  0.858                           |


### Softmax + GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               |                   |                |   0.928                       |  0.982                      |
| Layer 2               |                   |                |   0.932                       |  0.948                      |
| Layer 3               |                   |                |   0.929                       |  0.960                      |
| Layer 4               |                   |                |   0.929                       |  0.940                      |
| Layer 5               |                   |                |   0.928                       |  0.949                      |
| Layer 6               |                   |                |   0.930                       |  0.948                      |



---


### ➤ SN Mod (RT-DETR with Spectral Norm) Baseline mAP: 0.520

#### RT-DETR Logits (mAP in parenthesis)
| Method                  | AUROC           | Temp-Scaled AUROC | AUROC (Pruned <20) | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|------------------|--------------------|----------------------|-------------------------------|
| Softmax                 |                  |                    |                      |                               |
| Logsumexp (density)     |                  |                    |                      |                               |
| Entropy (Energy)        |                  |                    |                      |                               |


### GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               |                   |                |                               |                             |
| Layer 2               |                   |                |                               |                             |
| Layer 3               |                   |                |                               |                             |
| Layer 4               |                   |                |                               |                             |
| Layer 5               |                   |                |                               |                             |
| Layer 6               |                   |                |                               |                             |
| Layer 1 (Temp-Scaled) |                   |                |                               |                             |
| Layer 2 (Temp-Scaled) |                   |                |                               |                             |
| Layer 3 (Temp-Scaled) |                   |                |                               |                             |
| Layer 4 (Temp-Scaled) |                   |                |                               |                             |
| Layer 5 (Temp-Scaled) |                   |                |                               |                             |
| Layer 6 (Temp-Scaled) |                   |                |                               |                             |



> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above) (mAP in parenthesis)
| Num Gaussians Per Class | Temp-Scaled AUROC | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|--------------------|----------------------------------|
| 2                       |                    |                                  |
| 3                       |                    |                                  |
| 4                       |                    |                                  |



### Softmax + GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               |                   |                |                               |                             |
| Layer 2               |                   |                |                               |                             |
| Layer 3               |                   |                |                               |                             |
| Layer 4               |                   |                |                               |                             |
| Layer 5               |                   |                |                               |                             |
| Layer 6               |                   |                |                               |                             |