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
| Softmax             |  0.695               |    0.699                         |
| Logsumexp (density) |  0.667               |    0.712                         |
| Entropy (Energy)    |  0.713               |    0.771                         |




### GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC (Pruned <20) | Entropy AUROC (Pruned <20) |
|-----------------------|-------------------------------|-----------------------------|
| Layer 1               |   0.715                       |   0.632                     |
| Layer 2               |   0.704                       |   0.650                     |
| Layer 3               |   0.689                       |   0.649                     |
| Layer 4               |   0.692                       |   0.645                     |
| Layer 5               |   0.677                       |   0.618                     |
| Layer 6               |   0.689                       |   0.620                     |
| Layer 1 (Temp-Scaled) |   0.711                       |   0.640                     |
| Layer 2 (Temp-Scaled) |   0.701                       |   0.658                     |
| Layer 3 (Temp-Scaled) |   0.687                       |   0.660                     |
| Layer 4 (Temp-Scaled) |   0.689                       |   0.668                     |
| Layer 5 (Temp-Scaled) |   0.670                       |   0.665                     |
| Layer 6 (Temp-Scaled) |   0.685                       |   0.670                     |




> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above) (mAP in parenthesis)
| Num Gaussians Per Class | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|----------------------------------|
| 2                       |   0.745                          |
| 3                       |   0.748                          |
| 4                       |   0.747                          |





### Softmax + GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC (Pruned <20) | Entropy AUROC (Pruned <20) |
|-----------------------|-------------------------------|-----------------------------|
| Layer 1               |   0.789                       |   0.723                     |
| Layer 2               |   0.772                       |   0.738                     |
| Layer 3               |   0.754                       |   0.732                     |
| Layer 4               |   0.763                       |   0.736                     |
| Layer 5               |   0.765                       |   0.729                     |
| Layer 6               |   0.763                       |   0.727                     |



---


### ➤ SN Mod (RT-DETR with Spectral Norm) Baseline mAP: 0.520



#### RT-DETR Logits (mAP in parenthesis)
| Method              | AUROC (Pruned <20) | Temp-Scaled AUROC (Pruned <20) |
|---------------------|----------------------|----------------------------------|
| Softmax             |   0.641              |   0.692                          |
| Logsumexp (density) |   0.597              |   0.686                          |
| Entropy (Energy)    |   0.663              |   0.740                          |




### GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC (Pruned <20) | Entropy AUROC (Pruned <20) |
|-----------------------|-------------------------------|-----------------------------|
| Layer 1               |   0.653                       |   0.644                     |
| Layer 2               |   0.650                       |   0.637                     |
| Layer 3               |   0.648                       |   0.628                     |
| Layer 4               |   0.674                       |   0.619                     |
| Layer 5               |   0.678                       |   0.626                     |
| Layer 6               |   0.679                       |   0.611                     |
| Layer 1 (Temp-Scaled) |   0.641                       |   0.647                     |
| Layer 2 (Temp-Scaled) |   0.639                       |   0.641                     |
| Layer 3 (Temp-Scaled) |   0.640                       |   0.637                     |
| Layer 4 (Temp-Scaled) |   0.668                       |   0.630                     |
| Layer 5 (Temp-Scaled) |   0.672                       |   0.642                     |
| Layer 6 (Temp-Scaled) |   0.676                       |   0.633                     |





> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above) (mAP in parenthesis)
| Num Gaussians Per Class | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|----------------------------------|
| 2                       |   0.710                          |
| 3                       |   0.715                          |
| 4                       |   0.714                          |




### Softmax + GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC (Pruned <20) | Entropy AUROC (Pruned <20) |
|-----------------------|-------------------------------|-----------------------------|
| Layer 1               |    0.716                      |   0.755                     |
| Layer 2               |    0.737                      |   0.736                     |
| Layer 3               |    0.737                      |   0.724                     |
| Layer 4               |    0.751                      |   0.718                     |
| Layer 5               |    0.761                      |   0.716                     |
| Layer 6               |    0.756                      |   0.720                     |
