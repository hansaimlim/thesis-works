
# Weighted, Imputed, Neighbor-regularized matrix TriFactorization, WINTF

 [U,S,V]=WINTF(gene_tf, gene_gene_sim, tf_tf_sim, parameters);
 Y_hat=U*S*V';     %note that Y_hat includes prediction scores for known pairs as well.
 Y_hat=Y_hat.*(~gene_tf);     %removes prediction scores for known pairs.
 ```

------

```
REFERENCES
[1] Hansaim Lim and Lei Xie. "A New Weighted Imputed Neighborhood-regularized Tri-factorization One-class Collaborative Filtering Algorithm: Application to Target Gene Prediction of Transcription Factors" IEEE/TCBB, 2019.
[2] Hansaim Lim and Lei Xie. “Target gene prediction of transcription factor using a new neighborhood-regularized tri-factorization one-class collaborative filtering algorithm” Proceedings in the 9th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (ACM BCB) (2018).
[3] Annie Wang, Hansaim Lim (co-first author), Shu-Yuan Cheng, and Lei Xie. “ANTENNA, a multi-rank, multi-layered recommender system for inferring reliable drug-gene-disease associations: repurposing diazoxide as a targeted anti-cancer therapy.” IEEE/ACM Transactions on Computational Biology and Bioinformatics (2018). DOI: 10.1109/TCBB.2018.2812189
```
