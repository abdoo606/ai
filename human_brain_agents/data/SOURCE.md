# Data source

- `hcp_structural_connectome_82.csv`: 82x82 group-average structural connectome
  (diffusion MRI tractography, Human Connectome Project, S1200), 68 cortical
  Desikan-Killiany regions plus 14 subcortical regions (incl. hippocampus,
  amygdala, thalamus, caudate, putamen, pallidum, accumbens). Symmetric matrix.
- `hcp_structural_labels_82.txt`: one region label per line, same order as the matrix.
- Source: `MICA-MNI/ENIGMA` repository, `enigmatoolbox/datasets/matrices/hcp_connectivity/`
  (files `strucMatrix_with_sctx.csv`, `strucLabels_with_sctx.csv`).
- License: BSD 3-Clause, see `ENIGMA_LICENSE.txt`.

The matrix is undirected and has no distances, so this model uses it as an
undirected coupling strength. Directions and delays are simplified.
