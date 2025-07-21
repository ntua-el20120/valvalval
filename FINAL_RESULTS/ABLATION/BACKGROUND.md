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
| Method                  | AUROC           | Temp-Scaled AUROC | AUROC (Pruned <20) | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|------------------|--------------------|----------------------|-------------------------------|
| Softmax                 |  0.280           |  0.280             |   0.949              |    0.911                      |
| Logsumexp (density)     |  0.260           |  0.262             |   0.930              |    0.906                      |
| Entropy (Energy)        |  0.365           |  0.365             |   0.926              |    0.934                      |



### GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               |  0.429            |  0.543*        |    0.933                      |  0.544*                     |
| Layer 2               |  0.470            |  0.569*        |    0.944                      |  0.561*                    |
| Layer 3               |  0.369            |  0.574*        |    0.892                      |  0.578*                     |
| Layer 4               |  0.338            |  0.523*        |    0.906                      |  0.516*                     |
| Layer 5               |  0.289            |  0.520*        |    0.885                      |  0.508*                    |
| Layer 6               |  0.251            |  0.538*        |    0.837                      |  0.510*                     |
| Layer 1 (Temp-Scaled) |  0.425            |  0.560         |    0.932                      |  0.932                      |
| Layer 2 (Temp-Scaled) |  0.463            |  0.643         |    0.944                      |  0.934                      |
| Layer 3 (Temp-Scaled) |  0.365            |  0.599         |    0.890                      |  0.954                      |
| Layer 4 (Temp-Scaled) |  0.337            |  0.685         |    0.906                      |  0.956                      |
| Layer 5 (Temp-Scaled) |  0.289            |  0.682         |    0.884                      |  0.969                      |
| Layer 6 (Temp-Scaled) |  0.251            |  0.753         |    0.834                      |  0.947                      |




> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above) (mAP in parenthesis)
| Num Gaussians Per Class | Temp-Scaled AUROC | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|--------------------|----------------------------------|
| 2                       |  0.756*            |    0.963                         |
| 3                       |  0.755*            |    0.929                         |
| 4                       |  0.754*            |    0.923                         |



### Softmax + GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               |  0.350            |   0.420        |    0.940                      |   0.950                     |
| Layer 2               |  0.383            |   0.464        |    0.945                      |   0.934                     |
| Layer 3               |  0.324            |   0.433        |    0.939                      |   0.941                     |
| Layer 4               |  0.306            |   0.475        |    0.941                      |   0.943                     |
| Layer 5               |  0.282            |   0.485        |    0.935                      |   0.954                     |
| Layer 6               |  0.261            |   0.532        |    0.930                      |   0.932                     |



---


### ➤ SN Mod (RT-DETR with Spectral Norm) Baseline mAP: 0.520





#### RT-DETR Logits (mAP in parenthesis)
| Method                  | AUROC           | Temp-Scaled AUROC | AUROC (Pruned <20) | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|------------------|--------------------|----------------------|-------------------------------|
| Softmax                 |  0.267           |  0.267             |   0.927              |  0.887                        |
| Logsumexp (density)     |  0.254           |  0.252             |   0.852              |  0.877                        |
| Entropy (Energy)        |  0.312           |  0.312             |  0.827               |  0.888                        |




### GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               |  0.404            |  0.530*        |    0.899                      |   0.616*                    |
| Layer 2               |  0.362            |  0.447*        |    0.850                      |   0.676*                    |
| Layer 3               |  0.334            |  0.538*        |     0.825                     |   0.569*                    |
| Layer 4               |  0.281            |  0.516*        |    0.789                      |   0.553*                    |
| Layer 5               |  0.278            |  0.508*        |    0.823                      |   0.522*                    |
| Layer 6               |  0.270            |  0.509*        |    0.787                      |   0.517*                    |
| Layer 1 (Temp-Scaled) |  0.403            |  0.538         |    0.897                      |   0.863                     |
| Layer 2 (Temp-Scaled) |  0.364            |  0.448         |    0.847                      |   0.889                     |
| Layer 3 (Temp-Scaled) |  0.333            |  0.656         |    0.824                      |   0.912                     |
| Layer 4 (Temp-Scaled) |  0.280            |  0.553         |    0.787                      |   0.908                     |
| Layer 5 (Temp-Scaled) |  0.278            |  0.682         |    0.821                      |   0.891                     |
| Layer 6 (Temp-Scaled) |  0.270            |  0.674         |    0.785                      |   0.892                     |





> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above) (mAP in parenthesis)
| Num Gaussians Per Class | Temp-Scaled AUROC | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|--------------------|----------------------------------|
| 2                       |  0.702             |      0.874                       |
| 3                       |  0.703             |      0.866                       |
| 4                       |  0.700*            |      0.806                       |



### Softmax + GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               |   0.333           |  0.395         |      0.919                    |     0.930                   |
| Layer 2               |   0.317           |  0.343         |      0.920                    |     0.917                   |
| Layer 3               |   0.301           |  0.456         |      0.918                    |     0.923                   |
| Layer 4               |   0.272           |  0.395         |      0.914                    |     0.922                   |
| Layer 5               |   0.271           |  0.458         |      0.907                    |     0.920                   |
| Layer 6               |   0.266           |  0.464         |      0.914                    |     0.912                   |
