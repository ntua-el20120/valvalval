Αποτελέσματα AUROC για το απλό RTDETR μοντέλου (χωρίς αρχιτεκτονικές αλλαγές) για διαφορετικούς υπολογισμούς του AUROC.

Για να υπολογίσουμε το AUROC στο image classification χρειαζόμαστε τα εξής:
    -Εικόνες με κλάση in distribution
    -Εικόνες με κλάση out of distribution
    -Μια συνάρτηση f που να δίνει βαθμολογεί την εικόνα 

Αν η f παίρνει μεγάλες τιμές για in distribution εικόνες και μικρές για ood εικόνες, ονομάζεται confidence, διαφορετικά uncertainty

Σημείωση: Στο DDU paper και γενικά στο image classification δεν μας ενδιαφέρει αν το μοντέλο έδωσε τη σωστή κλάση για τα id δεδομένα, αλλά μόνο το score της f
Παράδειγμα: Έχω ένα μοντέλο που παίρνει ως input τα δεδομένα και βγάζει ως output (κλάση, confidence). Για τον υπολογισμό του auroc, αν η εικόνα ήταν id, δεν μας νοιάζει το output.κλάση του μοντέλου αλλά μόνο το confidence. Συνεπώς για μια id εικόνα το output (λάθος κλάση, 100% certainty) είναι true positive

Πρόβλημα με υπολογισμό AUROC σε object detection:
    Στο object detection τα ground truths μπορούν να είναι id ή ood, αλλά δεν είναι σίγουρο ότι το μοντέλο θα βγάλει έξοδο με bbox πάνω στο ground truth. Μάλιστα είναι σίγουρο ότι η έξοδος του μοντέλου ΔΕΝ θα είναι 100% ίδια με το bbox του ground truth. Αυτό μας δημιουργεί τις εξής περιπτώσεις για μια έξοδο του μοντέλου:
    - Η έξοδος κάνει match με κάποιο ground truth (id ή ood)
    - H έξοδος δεν κάνει match με κανένα ground truth (background ή χειρότερα partially overlap με κάποιο bbox)

Το πρώτο πρόβλημα που δημιουργείται είναι "Tι κάνουμε με τα unmatched outputs?" 
Το δεύτερο πρόβλημα που δημιουργείται είναι "Πώς κάνουμε το matching?"

Για την πρώτη ερώτηση οι επιλογές είναι 2:
    - Αγνοούμε τελείως τα background detections
    - Τα θεωρούμε out of distribution detections (αφού το background είναι ood)

    Θετικά και αρνητικά των επιλογών:
        Δεδομένου ότι το rtdetr δίνει 300 outputs ανά εικόνα (σταθερά), τα περισσότερα outputs είναι ood οδηγώντας σε αχρείαστα πολλά ood outputs (φαίνεται στις γραφικές στον φάκελο nomod_background_nocorrectclass). Από την άλλη θέλουμε να σιγουρέψουμε ότι ο συνδυασμός (μοντέλο,f) δίνει χαμηλό confidence στα background detections.

Για την δεύτερη ερώτηση οι επιλογές είναι 2*:
    - Κάνουμε matching των detections στα ground truths χρησιμοποιώντας τον hungarian matcher του rtdetr modified για να μη δίνει κόστος στην κλάση (αφού κάποιες κλάσεις δεν τις αναγνωρίζει και κρασάρει)
    - Κάνουμε matching των detections στα ground truths χρησιμοποιώντας ένα overlap threshold (με αποτέλεσμα πολλά detections να μπορούν να κάνουν match στο ίδιο ground truth αφού δεν κάνουμε non maximum suppression). Αυτή την τεχνική (σχεδόν) χρησιμοποιεί το mmdet

Σημείωση για το openset detection paper που βασιζόμαστε:
    Στο openset detection, στα αποτελέσματά τους χρησιμοποιούν ως id scores μόνο τα detections που έκαναν match σε id ground truth ΚΑΙ είπαν τη σωστή κλάση. Αυτό, βάσει του ορισμού του AUROC πάει ενάντια στην πρώτη σημείωση και κάνει artificially inflate τα αποτελέσματα αφού τα λάθος id detections (με μεγαλύτερο uncertainty) αγνοούνται.

Αυτό μας οδηγεί σε 4 είδη detections:
    Overlap με id gt & σωστή κλάση: 0
    Overlap με id gt & λάθος κλάση: 1
    Overlap με ood gt:              2
    Όχι overlap με gt (backgroung): 3

Στο openset detection paper θεωρούν id τα 0 και ood τα 2 ενώ τα υπόλοιπα αγνοούνται


Μια τελευταία ερώτηση που προκύπτει είναι η εξής:
    Κάποιες από τις συναρτήσεις f που χρησιμοποιούμε έχουν gaussians ή gaussian mixtures που εκπαιδεύονται πάνω στα training data. Η ίδιες ερωτήσεις που κάναμε πριν για το matching ισχύουν και εδώ. Δηλαδή αν θα κάνουμε matching με τον Hungarian Matcher (εδώ μπορούμε να πάρουμε τον non modified) ή με τα Overlaps.

Δοκιμάζουμε διαφορετικούς συνδυασμούς των παραπάνω επιλογών και παρουσιάζουμε τα αποτελέσματα:


# Openset Detection methodology 
Κάνουμε matching με το max iou και λαμβάνουμε τα detections 0 ως id / 2 ως ood / αγνοούμε 1,3

Αποτελέσματα

RTDETR LOGITS
| Method                  | AUROC           | Temp-Scaled AUROC |
|-------------------------|------------------|--------------------|
| Softmax                 | 0.913            | 0.889              |
| Logsumexp (density)     | 0.889            | 0.897              |
| Entropy (Energy)        | 0.908            | 0.923              |

Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο nomod_nobackground_correctclass/plots

GMM TRAINED WITH HUNGARIAN MATCHER
| Layer     | Logsumexp AUROC | Entropy AUROC | Temp-Scaled Logsumexp AUROC | Temp-Scaled Entropy AUROC |
|-----------|------------------|----------------|------------------------------|----------------------------|
| Layer 1   | 0.756            | 0.815*         | 0.741                        | 0.887                      |
| Layer 2   | 0.763            | 0.829*         | 0.750                        | 0.894                      |
| Layer 3   | 0.793            | 0.828*         | 0.782                        | 0.895                      |
| Layer 4   | 0.819            | 0.797*         | 0.812                        | 0.895                      |
| Layer 5   | 0.804            | 0.767*         | 0.791                        | 0.892                      |
| Layer 6   | 0.862            | 0.755*         | 0.857                        | 0.890                      |

Results with asterisks had -Infinity as at least one of the tpr@osr thresholds (0.05, 0.1, 0.2) and thus can't be used in inference

Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο nomod_nobackground_correctclass/plots

GMM PER CLASS WITH HUNGARIAN MATCHER (layer and function from best temperature scaled score from above table: Layer 3 function entropy)
| Num Gaussians Per Class | Temp-Scaled AUROC |
|-------------------------|--------------------|
| 2                     | 0.884              |
| 3                     | 0.892              |
| 4                     | 0.891              |

Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο nomod_nobackground_correctclass/plots

GMM TRAINED WITH MAX IOU MATCHINGS  (correct class matchings lead to empty dataset)
| Layer     | Logsumexp AUROC | Entropy AUROC | Temp-Scaled Logsumexp AUROC | Temp-Scaled Entropy AUROC |
|-----------|------------------|----------------|------------------------------|----------------------------|
| Layer 1   |                  |                |                              |                            |
| Layer 2   |                  |                |                              |                            |
| Layer 3   |                  |                |                              |                            |
| Layer 4   |                  |                |                              |                            |
| Layer 5   |                  |                |                              |                            |
| Layer 6   |                  |                |                              |                            |

Results with asterisks had -Infinity as at least one of the tpr@osr thresholds (0.05, 0.1, 0.2) and thus can't be used in inference

Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο nomod_nobackground_correctclass/plots

GMM PER CLASS WITH MAX IOU MATCHINGS (layer and function from best temperature scaled score from above table: Layer 3 function entropy)
| Num Gaussians Per Class | Temp-Scaled AUROC |
|-------------------------|--------------------|
| 2                     |              |
| 3                     |              |
| 4                     |              |

Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο nomod_nobackground_correctclass_train-no-matcher/plots


# Openset Detection Methodology without correct class checking
Κάνουμε matching με το max iou και λαμβάνουμε τα detections 0,1 ως id / 2 ως ood / αγνοούμε 3

Αποτελέσματα

RTDETR LOGITS
| Method                  | AUROC           | Temp-Scaled AUROC |
|-------------------------|------------------|--------------------|
| Softmax                 | 0.695            | 0.699              |
| Logsumexp (density)     | 0.667            | 0.719              |
| Entropy (Energy)        | 0.716            | 0.772              |

Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο nomod_nobackground_nocorrectclass/plots


GMM TRAINED WITH HUNGARIAN MATCHER
| Layer     | Logsumexp AUROC | Entropy AUROC | Temp-Scaled Logsumexp AUROC | Temp-Scaled Entropy AUROC |
|-----------|------------------|----------------|------------------------------|----------------------------|
| Layer 1   | 0.686                  | 0.636*               | 0.679                             | 0.651                           |
| Layer 2   | 0.678                 | 0.654*               | 0.673                             | 0.665                           |
| Layer 3   | 0.669                 | 0.648*               | 0.665                             | 0.667                           |
| Layer 4   | 0.684                 | 0.639*               | 0.682                             | 0.673                           |
| Layer 5   | 0.681                 | 0.623*               | 0.674                             | 0.674                           |
| Layer 6   | 0.699                 | 0.611*               | 0.696                             | 0.667                           |

Results with asterisks had -Infinity as at least one of the tpr@osr thresholds (0.05, 0.1, 0.2) and thus can't be used in inference

Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο nomod_nobackground_nocorrectclass/plots

GMM PER CLASS WITH HUNGARIAN MATCHER (layer and function from best temperature scaled score from above table: Layer 6 function logsumexp)
| Num Gaussians Per Class | Temp-Scaled AUROC  |
|-------------------------|--------------------|
| 2                       | 0.725              |
| 3                       | 0.728              |
| 4                       | 0.733              |

Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο nomod_nobackground_nocorrectclass/plots

GMM TRAINED WITH MAX IOU MATCHINGS  
| Layer     | Logsumexp AUROC | Entropy AUROC | Temp-Scaled Logsumexp AUROC | Temp-Scaled Entropy AUROC |
|-----------|------------------|----------------|------------------------------|----------------------------|
| Layer 1   |                  |                |                              |                            |
| Layer 2   |                  |                |                              |                            |
| Layer 3   |                  |                |                              |                            |
| Layer 4   |                  |                |                              |                            |
| Layer 5   |                  |                |                              |                            |
| Layer 6   |                  |                |                              |                            |

Results with asterisks had -Infinity as at least one of the tpr@osr thresholds (0.05, 0.1, 0.2) and thus can't be used in inference

Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο nomod_nobackground_nocorrectclass_train-no-matcher/plots

GMM PER CLASS WITH MAX IOU MATCHINGS (layer and function from best temperature scaled score from above table: Layer 3 function entropy)
| Num Gaussians Per Class | Temp-Scaled AUROC |
|-------------------------|--------------------|
| 2                     |              |
| 3                     |              |
| 4                     |              |

Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο nomod_nobackground_nocorrectclass_train-no-matcher/plots


# Openset Detection Methodology without correct class checking and background included
Κάνουμε matching με το max iou και λαμβάνουμε τα detections 0,1 ως id / 2,3 ως ood

Αποτελέσματα

RTDETR LOGITS  
| Method                  | AUROC           | Temp-Scaled AUROC |
|-------------------------|------------------|--------------------|
| Softmax                 | 0.684            | 0.686              |
| Logsumexp (density)     | 0.631            | 0.657              |
| Entropy (Energy)        | 0.622            | 0.647              |


Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο /plots


GMM TRAINED WITH HUNGARIAN MATCHER  
| Layer     | Logsumexp AUROC | Entropy AUROC | Temp-Scaled Logsumexp AUROC | Temp-Scaled Entropy AUROC |
|-----------|------------------|----------------|------------------------------|----------------------------|
| Layer 1   | 0.468            | 0.629*         | 0.457                        | 0.651                      |
| Layer 2   | 0.476            | 0.619*         | 0.466                        | 0.638                      |
| Layer 3   | 0.497            | 0.604*         | 0.489                        | 0.627                      |
| Layer 4   | 0.506            | 0.596*         | 0.499                        | 0.626                      |
| Layer 5   | 0.490            | 0.586*         | 0.480                        | 0.631                      |
| Layer 6   | 0.526            | 0.575*         | 0.520                        | 0.629                      |


Results with asterisks had -Infinity as at least one of the tpr@osr thresholds (0.05, 0.1, 0.2) and thus can't be used in inference

Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο /plots

GMM PER CLASS WITH HUNGARIAN MATCHER (layer and function from best temperature scaled score from above table: Layer 1 function entropy )
| Num Gaussians Per Class | Temp-Scaled AUROC |
|-------------------------|--------------------|
| 2                       | 0.647              |
| 3                       | 0.650              |
| 4                       | 0.641              |


Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο /plots


# Hungarian Matcher Matching with correct class checking
Κάνουμε matching με τοn modified hungarian matcher και λαμβάνουμε τα detections 0,1 ως id / 2 ως ood / αγνοούμε 3

Αποτελέσματα

RTDETR LOGITS  
| Method                  | AUROC           | Temp-Scaled AUROC |
|-------------------------|------------------|--------------------|
| Softmax                 |                  |                    |
| Logsumexp (density)     |                  |                    |
| Entropy (Energy)        |                  |                    |


Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο 


GMM TRAINED WITH HUNGARIAN MATCHER  
Note: Training detections for the GMMs check correct class too
| Layer     | Logsumexp AUROC | Entropy AUROC | Temp-Scaled Logsumexp AUROC | Temp-Scaled Entropy AUROC |
|-----------|------------------|----------------|------------------------------|----------------------------|
| Layer 1   |                  |                |                              |                            |
| Layer 2   |                  |                |                              |                            |
| Layer 3   |                  |                |                              |                            |
| Layer 4   |                  |                |                              |                            |
| Layer 5   |                  |                |                              |                            |
| Layer 6   |                  |                |                              |                            |


Results with asterisks had -Infinity as at least one of the tpr@osr thresholds (0.05, 0.1, 0.2) and thus can't be used in inference

Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο 

GMM PER CLASS WITH HUNGARIAN MATCHER (layer and function from best temperature scaled score from above table: )
| Num Gaussians Per Class | Temp-Scaled AUROC |
|-------------------------|--------------------|
| 2                       |                    |
| 3                       |                    |
| 4                       |                    |

Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο 

# Hungarian Matcher Mathcing without correct class checking
Κάνουμε matching με τοn modified hungarian matcher και λαμβάνουμε τα detections 0,1 ως id / 2 ως ood / αγνοούμε 3

Αποτελέσματα

RTDETR LOGITS  
| Method                  | AUROC           | Temp-Scaled AUROC |
|-------------------------|------------------|--------------------|
| Softmax                 |                  |                    |
| Logsumexp (density)     |                  |                    |
| Entropy (Energy)        |                  |                    |


Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο 


GMM TRAINED WITH HUNGARIAN MATCHER  
| Layer     | Logsumexp AUROC | Entropy AUROC | Temp-Scaled Logsumexp AUROC | Temp-Scaled Entropy AUROC |
|-----------|------------------|----------------|------------------------------|----------------------------|
| Layer 1   |                  |                |                              |                            |
| Layer 2   |                  |                |                              |                            |
| Layer 3   |                  |                |                              |                            |
| Layer 4   |                  |                |                              |                            |
| Layer 5   |                  |                |                              |                            |
| Layer 6   |                  |                |                              |                            |


Results with asterisks had -Infinity as at least one of the tpr@osr thresholds (0.05, 0.1, 0.2) and thus can't be used in inference

Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο 

GMM PER CLASS WITH HUNGARIAN MATCHER (layer and function from best temperature scaled score from above table: )
| Num Gaussians Per Class | Temp-Scaled AUROC |
|-------------------------|--------------------|
| 2                       |                    |
| 3                       |                    |
| 4                       |                    |

Σημείωση: Τα διαγράμματα φαίνονται στον φάκελο 