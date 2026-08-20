# Methods Section Template
# prophage_lysis v2.0.0 — for manuscript preparation
# ============================================================================
# Fill in [BRACKETED] values with your actual run statistics from
# data/output/pipeline_summary.json and your lab's experimental data.
#
# Target journal format: Nature Methods / Bioinformatics / Briefings in Bioinformatics
# Word count (all sections): ~1200 words
# ============================================================================


## Computational Discovery Pipeline

### Overview

We developed prophage_lysis, an open-source Snakemake-based pipeline (v2.0.0;
https://github.com/your-lab/prophage_lysis) for phenotype-guided discovery of
prophage lysis module enzymes from phage genome sequences. The pipeline accepts
phage genome FASTAs from any source and produces a prioritized list of endolysin
candidates for experimental evaluation, together with their cognate holin and
spanin partners. The source code, configuration, and test suite are publicly
available under the MIT license.


### Input preparation

Prophage sequences were extracted from [N] Vibrio spp. isolates using [PHASTER /
manual curation / PhageTerm] and submitted directly to the pipeline as individual
FASTA files. The pipeline is source-agnostic and does not require prior knowledge
of prophage boundaries.


### Gene annotation (M01)

Phage genomes were annotated using Pharokka v[VERSION] [CITE Pharokka] with the
Pharokka database (downloaded [DATE]). Coding sequences (CDS) were extracted
from the resulting GenBank files, retaining both amino acid sequences and
nucleotide CDS sequences for downstream codon analysis.


### Three-track lysis module identification (M02)

Protein sequences were searched against Pfam-A (release [RELEASE]) [CITE Pfam]
using HMMER v3.3.2 hmmscan (E-value ≤ 1 × 10⁻⁵). Candidate lysis module
proteins were identified using three parallel classification tracks. Endolysins
were identified by the presence of catalytic Pfam domains (PF00959, PF01520,
PF04851, PF13743, PF11860, PF08991, PF13529, PF06737, PF03237) or by
Pharokka functional annotation keywords (endolysin, lysin, muramidase, amidase,
CHAP, peptidoglycan, transglycosylase). SAR (signal-arrest-release) endolysins
were distinguished from holins by the presence of a single N-terminal
transmembrane helix followed by a downstream catalytic domain [CITE Xu 2004].
Holins were identified by holin-specific Pfam domains (PF04531, PF06840,
PF16754, PF14288, PF05102, PF17941) or by transmembrane topology (1–4
predicted TM helices, length ≤ 250 aa) in proteins lacking endolysin catalytic
domains. Spanins were identified by spanin-specific Pfam domains (PF16614,
PF11551, PF03278) and classified as i-spanins or o-spanins based on predicted
topology. Genomically proximal holins and endolysins (within 10 ORFs in the
same genome) were linked into lysis modules.


### Expressibility pre-filtering — Gate 1 (M03)

Candidates were filtered by biologically appropriate expressibility criteria
prior to embedding to reduce downstream compute requirements by approximately
40%. Endolysins were filtered by molecular weight (≤ 70 kDa), GRAVY index
(≤ 0.1 for non-SAR endolysins), instability index (Guruprasad et al., 1990;
≤ 60), and codon adaptation index (CAI ≥ 0.55, computed from nucleotide CDS
sequences against the E. coli K-12 reference adaptiveness table of Sharp and
Li [1987]). Holins were required to have at least one predicted transmembrane
helix and molecular weight ≤ 25 kDa. Candidates failing two or more criteria
were eliminated; candidates failing one criterion were flagged as warnings but
retained. Of [N_TOTAL] candidates, [N_PASS] passed Gate 1
([N_FAIL] eliminated, [N_WARN] flagged).


### Protein language model embeddings (M04)

Surviving candidates were represented as 1280-dimensional vectors using the
ESM-2 protein language model (650M parameters, trained on 250 million UniRef50
sequences) [CITE Lin 2023]. Sequences were embedded using [the ESM Metagenomic
Atlas REST API / a local ESM-2 instance on GPU]. Embeddings capture sequence-
agnostic functional information and enable comparison of candidates with limited
sequence similarity to known endolysins.


### Functional clustering (M05)

Embeddings were reduced to 50 dimensions using UMAP [CITE McInnes 2018]
(n_neighbors=15, min_dist=0.1, cosine metric) and clustered using HDBSCAN
[CITE Campello 2013] (min_cluster_size=3). Clustering was performed
independently for each functional track to prevent cross-track contamination of
cluster assignments. Noise points were assigned to the nearest cluster centroid
by cosine similarity. A separate 2D UMAP projection was computed for
visualization. This analysis identified [N_CLUSTERS_ENDO] endolysin clusters,
[N_CLUSTERS_HOLIN] holin clusters, and [N_CLUSTERS_SPANIN] spanin clusters.


### Peptidoglycan compatibility scoring (M06) [OPTIONAL — INCLUDE IF ENABLED]

Endolysins were scored for predicted activity against each target pathogen based
on catalytic domain type and published peptidoglycan chemistry compatibility
[CITE Gutierrez 2018]. Gram-negative pathogens (Vibrio harveyi, V.
parahaemolyticus, Edwardsiella tarda, Aeromonas hydrophila) were assigned the
DAP-type outer-membrane barrier chemotype; Streptococcus parauberis was assigned
the Lys-type Gram-positive chemotype. Scores of 2 (high probability), 1 (low
probability), or 0 (incompatible) were assigned based on known substrate ranges
of each catalytic domain type.


### Redundancy collapse — Gate 3 (M07)

Redundancy within each functional track was collapsed by clustering candidates
with pairwise cosine similarity > 0.92 in ESM-2 embedding space, corresponding
approximately to ≥ 85% amino acid sequence identity. For candidate pools larger
than 5000, similarity was computed using a block algorithm to avoid memory
overflow. A representative candidate was selected from each redundancy cluster
using a composite criterion weighting module completeness (holin + endolysin +
spanin co-occurrence), codon adaptation index, physicochemical expressibility,
and number of Gate 1 flags. This collapsed [N_BEFORE_G3] endolysin
representatives to [N_AFTER_G3].


### MaxMin diversity selection (M08)

Final candidates for expression were selected using the MaxMin (Farthest Point
Sampling) algorithm in ESM-2 embedding space [CITE Sener 2007], which greedily
maximizes the minimum pairwise cosine distance within the selected set. Selection
proceeded until the marginal diversity gain dropped below 30% of the initial
gain (saturation threshold), subject to minimum (n=5) and maximum (n=200)
constraints. For each selected endolysin, cognate holin and spanin from the same
lysis module were automatically co-selected. Selection was implemented using
Numba-compiled parallel CPU kernels. This yielded [N_SELECTED] endolysins
([N_COMPLETE_MOD] with complete holin+spanin modules) plus [N_HOLINS]
holins and [N_SPANINS] spanins, for [N_TOTAL_PRIORITY] total priority
candidates. Selected endolysins were queried against UniProtKB/Swiss-Prot via
NCBI BLAST (E-value ≤ 1 × 10⁻¹⁰); candidates with < 90% identity to any known
protein were classified as novel ([N_NOVEL]/[N_SELECTED]).


### Active learning re-ranking (M10–M11)

Following wet lab validation of round 1 candidates, binary activity labels
(active: MIC < 50 μg/mL or > 50% killing at 6 h; inactive otherwise) were used
to train per-pathogen activity classifiers on ESM-2 embedding features. For
[N_LABELED] labeled endolysins, [logistic regression / random forest] classifiers
were trained for each target pathogen (5-fold cross-validation ROC-AUC:
[AUC_VIB_H] for V. harveyi, [AUC_STREP] for S. parauberis, etc.).
Reserve candidates were scored by each classifier and ranked by mean predicted
activity probability across all pathogens, producing a re-ranked candidate list
for round 2 expression.


### Software and reproducibility

The complete pipeline is implemented in Python 3.10 using Snakemake v[VERSION]
for workflow management, BioPython v[VERSION] for sequence processing,
UMAP-learn v[VERSION] and HDBSCAN v[VERSION] for clustering, scikit-learn
v[VERSION] for active learning classifiers, and Numba v[VERSION] for JIT-
compiled distance kernels. All pipeline parameters are declared in a single
YAML configuration file. The pipeline produces a complete audit trail
(eliminated_log.csv) recording the elimination gate and reason for every
candidate. The source code, conda environment specification, and test suite are
available at https://github.com/your-lab/prophage_lysis (DOI: [ZENODO DOI]).


### Statistical analysis

[Add your wet lab statistical analysis methods here — MIC, time-kill,
ex vivo fish tissue model, etc.]
