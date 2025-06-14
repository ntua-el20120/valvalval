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
- We evaluate **unmodified RT-DETR** and **RT-DETR with Spectral Normalization (SN)**
- For each model, we compute AUROC:
  - Without and with **temperature scaling**
  - Without and with **pruning detections with Softmax < 20%**
- When pruning improves results, we also report the **new mAP**

---

## 🔹 Definition 1 – Correct Class Mapping

### ➤ No Mod (Original RT-DETR)

#### RT-DETR Logits (mAP in parenthesis)
| Method                  | AUROC           | Temp-Scaled AUROC | AUROC (Pruned <20) | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|------------------|--------------------|----------------------|-------------------------------|
| Softmax                 | 0.080            | 0.080              | 0.875                | 0.842                         |
| Logsumexp (density)     | 0.069            | 0.071              | 0.870                | 0.855                         |
| Entropy (Energy)        | 0.310            | 0.310              | 0.927                | 0.939                         |


### GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               | 0.466             | 0.591*         | 0.913                         | 0.575*                      |
| Layer 2               | 0.499             | 0.572*         | 0.917                         | 0.593*                      |
| Layer 3               | 0.440             | 0.468*         | 0.907                         | 0.606*                      |
| Layer 4               | 0.441             | 0.541*         | 0.926                         | 0.527*                      |
| Layer 5               | 0.262             | 0.509*         | 0.910                         | 0.514*                      |
| Layer 6               | 0.104             | 0.508*         | 0.874                         | 0.515*                      |
| Layer 1 (Temp-Scaled) | 0.461             | 0.594          | 0.910                         | 0.935                       |
| Layer 2 (Temp-Scaled) | 0.495             | 0.604          | 0.915                         | 0.927                       |
| Layer 3 (Temp-Scaled) | 0.442             | 0.472          | 0.906                         | 0.931                       |
| Layer 4 (Temp-Scaled) | 0.438             | 0.797          | 0.925                         | 0.932                       |
| Layer 5 (Temp-Scaled) | 0.261             | 0.679          | 0.910                         | 0.940                       |
| Layer 6 (Temp-Scaled) | 0.104             | 0.895          | 0.872                         | 0.894                       |



> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above) (mAP in parenthesis)
| Num Gaussians Per Class | Temp-Scaled AUROC | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|--------------------|----------------------------------|
| 2                       | 0.902              | 0.922                            |
| 3                       | 0.911              | 0.900                            |
| 4                       | 0.888*             | 0.913                            |




---


### ➤ SN Mod (RT-DETR with Spectral Norm)

#### RT-DETR Logits (mAP in parenthesis)
| Method                  | AUROC           | Temp-Scaled AUROC | AUROC (Pruned <20) | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|------------------|--------------------|----------------------|-------------------------------|
| Softmax                 | 0.126            | 0.126              | 0.871                | 0.916                         |
| Logsumexp (density)     | 0.120            | 0.127              | 0.814                | 0.870                         |
| Entropy (Energy)        | 0.418            | 0.418              | 0.910                | 0.939                         |


### GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               | 0.413             | 0.631*         | 0.826                         | 0.741*                      |
| Layer 2               | 0.393             | 0.548*         | 0.834                         | 0.786*                      |
| Layer 3               | 0.378             | 0.557*         | 0.821                         | 0.651*                      |
| Layer 4               | 0.222             | 0.632*         | 0.833                         | 0.597*                      |
| Layer 5               | 0.240             | 0.518*         | 0.845                         | 0.584*                      |
| Layer 6               | 0.206             | 0.505*         | 0.842                         | 0.561*                      |
| Layer 1 (Temp-Scaled) | 0.409             | 0.643          | 0.811                         | 0.948                       |
| Layer 2 (Temp-Scaled) | 0.392             | 0.574          | 0.826                         | 0.920                       |
| Layer 3 (Temp-Scaled) | 0.377             | 0.733          | 0.816                         | 0.933                       |
| Layer 4 (Temp-Scaled) | 0.216             | 0.740          | 0.829                         | 0.926                       |
| Layer 5 (Temp-Scaled) | 0.240             | 0.849          | 0.841                         | 0.926                       |
| Layer 6 (Temp-Scaled) | 0.206             | 0.794          | 0.837                         | 0.919                       |



> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above) (mAP in parenthesis)
| Num Gaussians Per Class | Temp-Scaled AUROC | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|--------------------|----------------------------------|
| 2                       | 0.854              | 0.925                            |
| 3                       | 0.851              | 0.867                            |
| 4                       | 0.826              | 0.852                            |





---

## 🔹 Definition 2 – Incorrect Class Mapping

### ➤ No Mod (Original RT-DETR) 

#### RT-DETR Logits (mAP in parenthesis)

| Method                  | AUROC           | Temp-Scaled AUROC | AUROC (Pruned <20) | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|------------------|--------------------|----------------------|-------------------------------|
| Softmax                 | 0.666            | 0.666              | 0.935                | 0.866                         |
| Logsumexp (density)     | 0.666            | 0.668              | 0.945                | 0.900                         |
| Entropy (Energy)        | 0.762            | 0.762              | 0.985                | 0.976                         |


### GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               | 0.799             | 0.714*         | 0.947                         | 0.582*                      |
| Layer 2               | 0.801             | 0.669*         | 0.959                         | 0.598*                      |
| Layer 3               | 0.792             | 0.734*         | 0.957                         | 0.613*                      |
| Layer 4               | 0.802             | 0.533*         | 0.976                         | 0.529*                      |
| Layer 5               | 0.729             | 0.507*         | 0.967                         | 0.516*                      |
| Layer 6               | 0.666             | 0.507*         | 0.914                         | 0.520*                      |
| Layer 1 (Temp-Scaled) | 0.796             | 0.878          | 0.945                         | 0.954                       |
| Layer 2 (Temp-Scaled) | 0.800             | 0.846          | 0.958                         | 0.942                       |
| Layer 3 (Temp-Scaled) | 0.789             | 0.886          | 0.957                         | 0.964                       |
| Layer 4 (Temp-Scaled) | 0.801             | 0.941          | 0.976                         | 0.959                       |
| Layer 5 (Temp-Scaled) | 0.729             | 0.939          | 0.967                         | 0.978                       |
| Layer 6 (Temp-Scaled) | 0.665             | 0.962          | 0.913                         | 0.940*                      |



> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above) (mAP in parenthesis)
| Num Gaussians Per Class | Temp-Scaled AUROC | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|--------------------|----------------------------------|
| 2                       | 0.964              | 0.964                            |
| 3                       | 0.959              | 0.924*                           |
| 4                       | 0.950              | 0.938*                           |



---

### ➤ SN Mod (RT-DETR with Spectral Norm)

#### RT-DETR Logits (mAP in parenthesis)
| Method                  | AUROC           | Temp-Scaled AUROC | AUROC (Pruned <20) | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|------------------|--------------------|----------------------|-------------------------------|
| Softmax                 | 0.573            | 0.573              | 0.976                | 0.950                         |
| Logsumexp (density)     | 0.568            | 0.574              | 0.975                | 0.968                         |
| Entropy (Energy)        | 0.725            | 0.725              | 0.988                | 0.987                         |


### GMM per Layer (mAP in parenthesis)
| Layer                 | Logsumexp AUROC   | Entropy AUROC  | Logsumexp AUROC (Pruned <20)  | Entropy AUROC (Pruned <20)  |
|-----------------------|-------------------|----------------|-------------------------------|-----------------------------|
| Layer 1               | 0.697             | 0.755*         | 0.910                         | 0.756*                      |
| Layer 2               | 0.677             | 0.773*         | 0.910                         | 0.808*                      |
| Layer 3               | 0.688             | 0.661*         | 0.935                         | 0.666*                      |
| Layer 4               | 0.620             | 0.705*         | 0.946                         | 0.624*                      |
| Layer 5               | 0.645             | 0.547*         | 0.960                         | 0.576*                      |
| Layer 6               | 0.616             | 0.518*         | 0.929                         | 0.560*                      |
| Layer 1 (Temp-Scaled) | 0.694             | 0.806          | 0.898                         | 0.984                       |
| Layer 2 (Temp-Scaled) | 0.673             | 0.789          | 0.904                         | 0.989                       |
| Layer 3 (Temp-Scaled) | 0.685             | 0.857          | 0.932                         | 0.990                       |
| Layer 4 (Temp-Scaled) | 0.615             | 0.870          | 0.944                         | 0.990                       |
| Layer 5 (Temp-Scaled) | 0.645             | 0.907          | 0.958                         | 0.989                       |
| Layer 6 (Temp-Scaled) | 0.615             | 0.899          | 0.926                         | 0.984                       |



> Results marked with * had `-inf` at one or more TPR@OSR thresholds (0.05, 0.1, 0.2) and can't be used in inference.


#### GMM PER CLASS WITH HUNGARIAN MATCHER (Best Layer + Metric from Above) (mAP in parenthesis)
| Num Gaussians Per Class | Temp-Scaled AUROC | Temp-Scaled AUROC (Pruned <20) |
|-------------------------|--------------------|----------------------------------|
| 2                       | 0.901              | 0.987                            |
| 3                       | 0.893              | 0.983                            |
| 4                       | 0.866              | 0.982                            |