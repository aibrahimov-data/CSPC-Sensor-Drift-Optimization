# PCA fact sheet (descriptive analysis only)

Files: `src/pca_analysis.py`, `results/tables/pca_explained_variance.csv`,
`results/figures/pca_by_gas.png`, `results/figures/pca_by_batch.png`.

## Method

- StandardScaler and PCA (10 components) are fitted on the source batches
  1-5 only (1,716 observations, 128 sensor features).
- The same fitted transformation is applied to the future batches 6-10
  (3,775 observations). Future data does not influence the fit.
- Concentration is not used. Original sensor measurements are not modified
  and no observations are removed.

## Explained variance

| Component | Explained variance | Cumulative |
|---|---:|---:|
| PC1 | 40.1% | 40.1% |
| PC2 | 21.9% | 62.0% |
| PC3 | 9.3% | 71.3% |
| PC4 | 7.4% | 78.8% |
| PC5 | 4.8% | 83.5% |
| PC10 | 1.2% | 93.2% |

The first two components explain 62.0% of the source variance, so the 2D
figures show only part of the structure in the data.

## Observations from the figures

- **Source batches (1-5):** ethanol and ethylene occupy largely different
  regions of the PC1-PC2 plane (ethylene in the upper part along diagonal
  bands, ethanol in the lower part), with few overlapping points.
- **Future batches (6-10):** both gases occupy regions that differ from the
  source layout. Ethylene points appear in a group below the main ethylene
  band and in a sparse tail towards low PC2, regions that few source points
  occupy. Ethanol and ethylene points lie closer together in parts of the
  future panel than in the source panel.
- **By batch:** in the batch-coloured figure, batch 10 forms a separate group
  for both gases. For ethanol, batch 10 appears close to the earliest batches
  rather than to the batches immediately before it, so the apparent shift is
  not a simple increasing function of batch number. This agrees with the fact
  that batch IDs are not equally spaced in time.
- In both source and future batches the two gases approach each other at the
  right end of the PC1 axis.
- A few extreme points (PC1 below about -20) are present and were kept.

## Limitations

- PCA is a descriptive visualisation. Overlap of the two classes in two
  components does not mean they are indistinguishable; the classifiers use
  all 128 features.
- An apparent shift between batches does not prove sensor ageing. Other
  factors, such as experimental conditions and concentration distributions,
  may contribute and are examined separately in the concentration analysis.
- Descriptions of the figures are qualitative and were not tested
  statistically.
