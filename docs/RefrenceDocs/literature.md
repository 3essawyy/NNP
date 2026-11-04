# Literature survey notes (REQ-M8, T-014)

What each source contributes to **our** design. Full BibTeX: [`report/refs.bib`](../../report/refs.bib).

> **Verification status:** every entry below is a well-known publication listed from memory during S001/S002. Before submission a team member must open each one (Google Scholar / IEEE Xplore / EKB) and confirm authors, venue, year and pages in `refs.bib` (T-014). Do not add a citation you have not opened.

## 1. The HAR feature set and dataset family
| Key | Source | Takeaway for NNP |
|---|---|---|
| `anguita2013` | Anguita, Ghio, Oneto, Parra, Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN 2013. | Defines the 561 time/frequency features (body/gravity acceleration, jerk, gyro, magnitudes, FFT bands, angles) this data is derived from; features were normalized to [−1, 1] — our training set violates that range in 251 features, which exposed the calibration offset. Classic result: sitting vs standing is the hard pair. |
| `wang2019deep` | Wang, Chen, Hao, Peng, Hu. *Deep learning for sensor-based activity recognition: A survey.* Pattern Recognition Letters, 2019. | Deep models shine on **raw** windows; on hand-crafted 561-feature vectors classical kernels/linear models are competitive and much cheaper — supports our CPU-only, sub-millisecond choice. |

## 2. Device heterogeneity (the core problem)
| Key | Source | Takeaway |
|---|---|---|
| `stisen2015` | Stisen et al. *Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition.* ACM SenSys 2015. | Phones differ in sensor bias, gain and sampling; models trained on one device degrade on others. Per-device normalization/calibration is a first-line mitigation — exactly our per-device standardization. |
| `quinonero2009` | Quiñonero-Candela, Sugiyama, Schwaighofer, Lawrence (eds.). *Dataset Shift in Machine Learning.* MIT Press, 2009. | Vocabulary: covariate shift vs prior shift. Our per-domain standardization assumes covariate (affine) shift and similar class priors across devices — stated as a limitation. |
| `moreno2012` | Moreno-Torres, Raeder, Alaiz-Rodríguez, Chawla, Herrera. *A unifying view on dataset shift in classification.* Pattern Recognition, 2012. | Taxonomy used in the report's problem statement. |
| `bendavid2010` | Ben-David, Blitzer, Crammer, Kulesza, Pereira, Vaughan. *A theory of learning from different domains.* Machine Learning, 2010. | Target error ≤ source error + domain divergence + λ: reducing divergence (aligning feature distributions) is what lowers target error — motivates alignment instead of only a stronger classifier. |

## 3. Domain generalization / adaptation methods
| Key | Source | Takeaway |
|---|---|---|
| `zhou2022dg` | Zhou, Liu, Qiao, Xiang, Loy. *Domain Generalization: A Survey.* IEEE TPAMI, 2022. | Map of DG families (alignment, augmentation, meta-learning, ensembles). Our simulated-shift augmentation for *evaluation* and alignment by normalization come from here. |
| `wang2022dg` | Wang, Lan, Liu, Ouyang, Qin, Lu, Chen, Zeng, Yu. *Generalizing to Unseen Domains: A Survey on Domain Generalization.* IEEE TKDE, 2022. | Same; includes sensor-based HAR examples. |
| `li2017adabn` | Li, Wang, Shi, Liu, Hou. *Revisiting Batch Normalization for Practical Domain Adaptation.* ICLR Workshop 2017 (arXiv:1603.04779). | AdaBN: re-estimate normalization statistics on the target domain, keep the weights. Our method is the non-deep analogue: per-device standardization with target statistics. |
| `sun2016coral` | Sun, Feng, Saenko. *Return of Frustratingly Easy Domain Adaptation* (CORAL). AAAI 2016. | Second-order alignment (covariances). Considered as future work; first-order per-feature alignment already closes most of the gap here. |

## 4. Classifiers, selection, robust statistics
| Key | Source | Takeaway |
|---|---|---|
| `cortes1995` | Cortes, Vapnik. *Support-vector networks.* Machine Learning, 1995. | RBF SVM (shipped model). |
| `chang2011libsvm` | Chang, Lin. *LIBSVM: A library for support vector machines.* ACM TIST, 2011. | One-vs-one multi-class decision we re-implement in numpy for speed. |
| `hsu2002` | Hsu, Lin. *A comparison of methods for multiclass support vector machines.* IEEE TNN, 2002. | Justifies one-vs-one. |
| `guyon2003` | Guyon, Elisseeff. *An introduction to variable and feature selection.* JMLR, 2003. | Filter methods (ANOVA F, MI) vs wrappers; we use a univariate filter fit on training data only. |
| `ledoit2004` | Ledoit, Wolf. *A well-conditioned estimator for large-dimensional covariance matrices.* J. Multivariate Analysis, 2004. | Shrinkage used by our LDA/QDA baselines. |
| `huber1981` | Huber. *Robust Statistics.* Wiley, 1981. | Median/IQR scaling and clipping (winsorizing) against outliers. |
| `rousseeuw1987` | Rousseeuw. *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.* J. Comput. Appl. Math., 1987. | Silhouette criterion for accepting a device split. |
| `ke2017lightgbm` | Ke et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017. | Gradient-boosting baseline. |
| `pedregosa2011` | Pedregosa et al. *Scikit-learn: Machine Learning in Python.* JMLR, 2011. | Implementation library. |
