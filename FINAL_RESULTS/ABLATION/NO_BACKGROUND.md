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
| Softmax                 |                  |                    |   0.875              |   0.842                       |
| Logsumexp (density)     |                  |                    |   0.870              |   0.855                       |
| Entropy (Energy)        |                  |                    |   0.927              |   0.939                       |


### GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               |                   |                |     0.912                     |     0.585*                  |
| Layer 2               |                   |                |     0.918                     |     0.594*                  |
| Layer 3               |                   |                |     0.907                     |     0.601*                  |
| Layer 4               |                   |                |     0.924                     |     0.530*                  |
| Layer 5               |                   |                |     0.912                     |     0.518*                  |
| Layer 6               |                   |                |     0.873                     |     0.519*                  |
| Layer 1 (Temp-Scaled) |                   |                |     0.909                     |     0.901                   |
| Layer 2 (Temp-Scaled) |                   |                |     0.916                     |     0.893                   |
| Layer 3 (Temp-Scaled) |                   |                |     0.906                     |     0.909                   |
| Layer 4 (Temp-Scaled) |                   |                |     0.924                     |     0.899                   |
| Layer 5 (Temp-Scaled) |                   |                |     0.912                     |     0.924                   |
| Layer 6 (Temp-Scaled) |                   |                |     0.872                     |     0.897                   |



> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above) (mAP in parenthesis)
| Num Gaussians Per Class | Temp-Scaled AUROC | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|--------------------|----------------------------------|
| 2                       |                    |   0.927                          |
| 3                       |                    |   0.856*                         |
| 4                       |                    |   0.875                          |



### Softmax + GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               |                   |                |    0.923                      |    0.915                    |
| Layer 2               |                   |                |    0.929                      |    0.891                    |
| Layer 3               |                   |                |    0.928                      |    0.908                    |
| Layer 4               |                   |                |    0.932                      |    0.910                    |
| Layer 5               |                   |                |    0.925                      |    0.929                    |
| Layer 6               |                   |                |    0.916                      |    0.915                    |


---


### ➤ SN Mod (RT-DETR with Spectral Norm) Baseline mAP: 0.520



#### RT-DETR Logits (mAP in parenthesis)
| Method                  | AUROC           | Temp-Scaled AUROC | AUROC (Pruned <20) | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|------------------|--------------------|----------------------|-------------------------------|
| Softmax                 |   0.126          |    0.126           | 0.871 (0.519)        |   0.916                       |
| Logsumexp (density)     |   0.120          |    0.127           | 0.814                |   0.870                       |
| Entropy (Energy)        |   0.418          |    0.418           | 0.910                |   0.939                       |



### GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               |  0.414            |   0.495*       |  0.826                        |   0.751*                    |
| Layer 2               |  0.390            |   0.637*       |  0.835                        |   0.788*                    |
| Layer 3               |  0.382            |   0.596*       |  0.822                        |   0.659*                    |
| Layer 4               |  0.218            |   0.628*       |  0.833                        |   0.616*                    |
| Layer 5               |  0.227            |   0.520*       |  0.845                        |   0.578*                    |
| Layer 6               |  0.196            |   0.510*       |  0.840                        |   0.564*                    |
| Layer 1 (Temp-Scaled) |  0.415            |   0.497        |  0.815                        |   0.952                     |
| Layer 2 (Temp-Scaled) |  0.384            |   0.637        |  0.827                        |   0.925                     |
| Layer 3 (Temp-Scaled) |  0.378            |   0.727        |  0.817                        |   0.937                     |
| Layer 4 (Temp-Scaled) |  0.211            |   0.699        |  0.830                        |   0.920                     |
| Layer 5 (Temp-Scaled) |  0.227            |   0.849        |  0.841                        |   0.927                     |
| Layer 6 (Temp-Scaled) |  0.196            |   0.756        |  0.836                        |   0.924                     |




> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above) (mAP in parenthesis)
| Num Gaussians Per Class | Temp-Scaled AUROC | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|--------------------|----------------------------------|
| 2                       |  0.858             |  0.936                           |
| 3                       |  0.856             |  0.857                           |
| 4                       |  0.826             |  0.858                           |


### Softmax + GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               |  0.358            |  0.407         |   0.928                       |  0.982                      |
| Layer 2               |  0.333            |  0.504         |   0.932                       |  0.948                      |
| Layer 3               |  0.327            |  0.562         |   0.929                       |  0.960                      |
| Layer 4               |  0.192            |  0.547         |   0.929                       |  0.940                      |
| Layer 5               |  0.206            |  0.651         |   0.928                       |  0.949                      |
| Layer 6               |  0.185            |  0.578         |   0.930                       |  0.948                      |