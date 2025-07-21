# Algorithm Comparison for AUROC Calculation on REAL FLIGHTS
We evaluate AUROC using two definitions:

- **Definition 1 (Correct):**  
  - ID Classes: `0,1`  
  - OOD Class: `2`  
  - Ignored Class: `3`


- We evaluate **unmodified RT-DETR** and **RT-DETR with Spectral Normalization (SN)**
- For each model, we compute AUROC:
  - Without and with **temperature scaling**
- When pruning improves results, we also report the **new mAP**

---

## 🔹 Definition 1 – Correct Class Mapping

### ➤ No Mod (Original RT-DETR) Baseline mAP: 0.542



#### RT-DETR Logits (mAP in parenthesis)
| Method              | AUROC (Pruned <20) | Temp-Scaled AUROC (Pruned <20) |
|---------------------|----------------------|----------------------------------|
| Softmax             |  0.848               |  0.854                           |
| Logsumexp (density) |  0.810               |  0.816                           |
| Entropy (Energy)    |  0.796               |  0.798                           |



### GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC (Pruned <20) | Entropy AUROC (Pruned <20) |
|-----------------------|-------------------------------|-----------------------------|
| Layer 1               |    0.748                      |   0.579*                    |
| Layer 2               |    0.785                      |   0.591*                    |
| Layer 3               |    0.782                      |   0.565*                    |
| Layer 4               |    0.779                      |   0.547*                    |
| Layer 5               |    0.751                      |   0.537*                    |
| Layer 6               |    0.683                      |   0.548*                    |
| Layer 1 (Temp-Scaled) |    0.746                      |   0.735                     |
| Layer 2 (Temp-Scaled) |    0.783                      |   0.749                     |
| Layer 3 (Temp-Scaled) |    0.780                      |   0.805                     |
| Layer 4 (Temp-Scaled) |    0.778                      |   0.818                     |
| Layer 5 (Temp-Scaled) |    0.750                      |   0.817                     |
| Layer 6 (Temp-Scaled) |    0.681                      |   0.816                     |




> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above) (mAP in parenthesis)
| Num Gaussians Per Class | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|----------------------------------|
| 2                       |   0.814                          |
| 3                       |   0.812                          |
| 4                       |   0.814                          |




### Softmax + GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC (Pruned <20) | Entropy AUROC (Pruned <20) |
|-----------------------|-------------------------------|-----------------------------|
| Layer 1               |     0.858                     |    0.856                    |
| Layer 2               |     0.856                     |    0.857                    |
| Layer 3               |     0.855                     |    0.859                    |
| Layer 4               |     0.855                     |    0.859                    |
| Layer 5               |     0.855                     |    0.858                    |
| Layer 6               |     0.855                     |    0.857                    |


---


### ➤ SN Mod (RT-DETR with Spectral Norm) Baseline mAP: 0.520



#### RT-DETR Logits (mAP in parenthesis)
| Method              | AUROC (Pruned <20) | Temp-Scaled AUROC (Pruned <20) |
|---------------------|----------------------|----------------------------------|
| Softmax             |    0.854             |  0.866                           |
| Logsumexp (density) |    0.789             |  0.799                           |
| Entropy (Energy)    |    0.771             |  0.774                           |




### GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC (Pruned <20) | Entropy AUROC (Pruned <20) |
|-----------------------|-------------------------------|-----------------------------|
| Layer 1               |   0.787                       |   0.538*                    |
| Layer 2               |   0.775                       |   0.550*                    |
| Layer 3               |   0.764                       |   0.546*                    |
| Layer 4               |   0.738                       |   0.500*                    |
| Layer 5               |   0.748                       |   0.521*                    |
| Layer 6               |   0.698                       |   0.550*                    |
| Layer 1 (Temp-Scaled) |   0.788                       |   0.727                     |
| Layer 2 (Temp-Scaled) |   0.775                       |   0.706                     |
| Layer 3 (Temp-Scaled) |   0.763                       |   0.751                     |
| Layer 4 (Temp-Scaled) |   0.738                       |   0.759                     |
| Layer 5 (Temp-Scaled) |   0.748                       |   0.806                     |
| Layer 6 (Temp-Scaled) |   0.697                       |   0.792                     |





> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above) (mAP in parenthesis)
| Num Gaussians Per Class | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|----------------------------------|
| 2                       |  0.790                           |
| 3                       |  0.769                           |
| 4                       |  0.781                           |



### Softmax + GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC (Pruned <20) | Entropy AUROC (Pruned <20) |
|-----------------------|-------------------------------|-----------------------------|
| Layer 1               |   0.874                       |   0.874                     |
| Layer 2               |   0.873                       |   0.871                     |
| Layer 3               |   0.871                       |   0.873                     |
| Layer 4               |   0.869                       |   0.868                     |
| Layer 5               |   0.868                       |   0.872                     |
| Layer 6               |   0.868                       |   0.874                     |