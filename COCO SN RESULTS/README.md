# Algorithm Comparison for AUROC Calculation on AOT

We evaluate AUROC using two definitions:

- **Definition 1 (Correct):**  
  - ID Classes: `0,1`  
  - OOD Class: `2`  
  - Ignored Class: `3`

- **Definition 2 (Incorrect):**  
  - ID Class: `0`  
  - OOD Class: `2`  
  - Ignored Classes: `1,3`  

For each definition:
- We evaluate  **RT-DETR with Spectral Normalization (SN)**
- For each model, we compute AUROC:
  - Without and with **temperature scaling**
  - Without and with **pruning detections with Softmax < 20%**
- When pruning improves results, we also report the **new mAP**

---

## 🔹 Definition 1 – Correct Class Mapping

### ➤ No Mod (Original RT-DETR)




### ➤ SN Mod (RT-DETR with Spectral Norm)

#### RT-DETR Logits 
| Method                  | AUROC           | Temp-Scaled AUROC | AUROC (Pruned <20) | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|------------------|--------------------|----------------------|-------------------------------|
| Softmax                 | 0.427            | 0.427              | 0.641                | 0.692                         |
| Logsumexp (density)     | 0.461            | 0.499              | 0.590                | 0.684                         |
| Entropy (Energy)        | 0.560            | 0.587              | 0.664                | 0.737                         |


### GMM per Layer
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               | 0.518             | 0.538          | 0.625                         | 0.639                       |
| Layer 2               | 0.522             | 0.526          | 0.632                         | 0.640                       |
| Layer 3               | 0.553             | 0.521          | 0.638                         | 0.624                       |
| Layer 4               | 0.552             | 0.525          | 0.663                         | 0.617*                      |
| Layer 5               | 0.574             | 0.543          | 0.676                         | 0.612*                      |
| Layer 6               | 0.576             | 0.542*         | 0.683                         | 0.595*                      |
| Layer 1 (Temp-Scaled) | 0.513             | 0.558          | 0.612                         | 0.650                       |
| Layer 2 (Temp-Scaled) | 0.519             | 0.535          | 0.620                         | 0.646                       |
| Layer 3 (Temp-Scaled) | 0.550             | 0.537          | 0.630                         | 0.635                       |
| Layer 4 (Temp-Scaled) | 0.548             | 0.545          | 0.657                         | 0.629                       |
| Layer 5 (Temp-Scaled) | 0.567             | 0.573          | 0.671                         | 0.637                       |
| Layer 6 (Temp-Scaled) | 0.572             | 0.571          | 0.681                         | 0.623                       |



> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above)
| Num Gaussians Per Class | Temp-Scaled AUROC | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|--------------------|----------------------------------|
| 2                       | 0.573              | 0.705                            |
| 3                       | 0.566              | 0.711                            |
| 4                       | 0.557              | 0.708                            |





---

## 🔹 Definition 2 – Incorrect Class Mapping


### ➤ SN Mod (RT-DETR with Spectral Norm)

#### RT-DETR Logits
| Method                  | AUROC           | Temp-Scaled AUROC | AUROC (Pruned <20) | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|------------------|--------------------|----------------------|-------------------------------|
| Softmax                 | 0.624            | 0.624              | 0.853                | 0.871                         |
| Logsumexp (density)     | 0.608            | 0.647              | 0.806                | 0.860                         |
| Entropy (Energy)        | 0.697            | 0.726              | 0.848                | 0.890                         |


### GMM per Layer
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               | 0.585             | 0.667          | 0.697                         | 0.813                       |
| Layer 2               | 0.588             | 0.646          | 0.704                         | 0.815                       |
| Layer 3               | 0.620             | 0.643          | 0.746                         | 0.807                       |
| Layer 4               | 0.634             | 0.642          | 0.791                         | 0.798*                      |
| Layer 5               | 0.673             | 0.658          | 0.814                         | 0.782*                      |
| Layer 6               | 0.688             | 0.666*         | 0.813                         | 0.758*                      |
| Layer 1 (Temp-Scaled) | 0.569             | 0.686          | 0.671                         | 0.836                       |
| Layer 2 (Temp-Scaled) | 0.574             | 0.656          | 0.679                         | 0.836                       |
| Layer 3 (Temp-Scaled) | 0.608             | 0.655          | 0.726                         | 0.830                       |
| Layer 4 (Temp-Scaled) | 0.624             | 0.661          | 0.779                         | 0.826                       |
| Layer 5 (Temp-Scaled) | 0.664             | 0.690          | 0.804                         | 0.834                       |
| Layer 6 (Temp-Scaled) | 0.681             | 0.705          | 0.807                         | 0.824                       |



> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above)
| Num Gaussians Per Class | Temp-Scaled AUROC | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|--------------------|----------------------------------|
| 2                       | 0.691              | 0.840                            |
| 3                       | 0.688              | 0.837                            |
| 4                       | 0.679              | 0.842                            |