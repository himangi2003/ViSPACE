# Colorectal cancer histopathology datasets

Public data for training classification, segmentation, nuclei and weakly supervised (multiple-instance, MIL) models, with annotation level, magnification and access.

Values marked ≈ or "check" are approximate. Verify them on the source page before using them in a paper.

---

## 1. Annotation levels

| Level | What is labelled | What it can train | Datasets |
|---|---|---|---|
| Slide label | One label per whole slide | Diagnosis, lesion grade, MSI and mutation status (via MIL) | TCGA COAD/READ, CPTAC-COAD, IMP-CRS 2024, PAIP 2020 (MSI) |
| Patch label | One label per tile/patch | Tissue-type and lesion classifiers; tile-level tissue maps | NCT-CRC-HE-100K, STARC-9, HunCRC, Kather 2016, UniToPatho, MHIST, Chaoyang, EBHI |
| Pixel / region mask | Class outlined per pixel | Semantic segmentation with real boundaries | SemiCOL, EBHI-Seg, PAIP 2020 (tumour), DigestPath 2019, GlaS, CRAG |
| Nuclei | Each nucleus outlined and typed | Cell detection/typing, e.g. lymphocytes for TIL scoring | Lizard, CoNIC, CoNSeP, CRCHisto, PanNuke |

---

## 2. Magnification

ViSPACE tiles are 224 px at 0.25 µm/px. Always read µm/px from the file metadata or paper.

| Magnification | µm per pixel | Datasets |
|---|---|---|
| 40x | ≈ 0.25 | STARC-9, HunCRC zoom_1 (≈0.24), IMP-CRS 2024, CoNSeP, PanNuke, PAIP 2020 WSIs, TCGA (mixed 20x/40x) |
| 20x | ≈ 0.5 | NCT-CRC-HE-100K, Kather 2016, HunCRC zoom_2 (≈0.48), UniToPatho (0.44), Chaoyang, GlaS, CRAG, Lizard, CoNIC, CRCHisto |
| 10x | ≈ 1.0 | SemiCOL annotated patches (as reported). Upsampling to 40x adds no detail, so train such data at 0.5–1.0 µm/px instead |

---

## 3. Tissue-type patch classification (patch labels)

| Dataset | What is annotated | Size | Scale | Access |
|---|---|---|---|---|
| NCT-CRC-HE-100K + CRC-VAL-HE-7K | 9 classes: adipose, background, debris, lymphocytes, mucus, muscle, normal mucosa, stroma, tumour | 100,000 + 7,180 tiles, 224 px; 86 + 50 slides | 0.5 µm/px (20x); Macenko-normalised and raw (NONORM) versions | Zenodo 1214456, open |
| STARC-9 (2025) | 9 classes: adipose, lymphoid, muscle, fibroconnective (stroma), mucin, necrosis, blood, tumour, normal mucosa | 630,000 tiles, 256 px; ~200 patients | 0.25 µm/px (40x) | Paper release (arXiv 2511.00383); check access |
| Kather 2016 | 8 classes: tumour, simple and complex stroma, lymphoid, debris, mucosa, adipose, empty | 5,000 tiles, 150 px | ≈0.5 µm/px (20x) | Zenodo 53169, open |
| HunCRC patches | 10 region classes incl. low/high-grade dysplasia, adenocarcinoma, necrosis, inflammation, normal | 200 slides; 512 px patches at 2 zooms | ≈0.24 and ≈0.48 µm/px | figshare collection 5927795, open |

---

## 4. Lesion and dysplasia classification (patch labels)

| Dataset | What is annotated | Size | Scale | Access |
|---|---|---|---|---|
| UniToPatho | Normal, hyperplastic polyp, tubular and tubulo-villous adenoma, each with low- or high-grade dysplasia (6 classes) | 9,536 patches from 292 WSIs | 0.44 µm/px (20x) | IEEE DataPort |
| MHIST | Hyperplastic polyp vs sessile serrated adenoma (majority of 7 pathologists) | 3,152 tiles, 224 px; 328 WSIs | Downsampled; check | Dartmouth, request form |
| Chaoyang | Normal, serrated, adenoma, adenocarcinoma; real label noise (3 pathologists) | 6,160 patches, 512 px | 20x | Authors' release |
| EBHI | Normal, polyp, low- and high-grade intraepithelial neoplasia, serrated adenoma, adenocarcinoma | 5,532 images | Several magnifications | figshare, open |

UniToPatho is the closest public match for a dysplasia-grade classifier.

---

## 5. Pixel and region segmentation (pixel masks)

| Dataset | What is annotated | Size | Scale | Access |
|---|---|---|---|---|
| SemiCOL 2023 | 11 labels: tumour, mucosa, tumour stroma, submucosa/fat, muscle, lymphoid, ulcer/necrosis, mucin, blood, background, unlabelled | 20 annotated WSIs (~72k patches) + 499 weakly labelled WSIs | 10x patches (reported) | semicol.org; registration and data agreement |
| EBHI-Seg | Lesion masks across 6 stages, normal to adenocarcinoma, incl. low/high-grade dysplasia | 4,456 images + masks | High magnification; check | figshare 21540159, open |
| PAIP 2020 | Whole-tumour-area masks plus MSI-H / MSS label (resections) | WSIs; val 31, test 40 (train: see site) | 40x WSIs | paip2020.grand-challenge.org; registration |
| DigestPath 2019 | Malignant lesion masks in colonoscopy biopsies; signet-ring cell boxes | Several hundred images | ≈20x | grand-challenge; registration |
| GlaS 2015 | Gland instance masks; benign vs malignant | 165 images | 20x | Warwick TIA, open |
| CRAG | Gland instance masks (adenocarcinoma) | 213 images | 20x | Warwick TIA, open |

SemiCOL is the best fit for a 5-class colorectal segmenter that includes stroma; it needs a data agreement.

---

## 6. Nuclei detection and classification

| Dataset | What is annotated | Size | Scale | Access |
|---|---|---|---|---|
| Lizard | 6 types: epithelial, lymphocyte, plasma cell, neutrophil, eosinophil, connective | ~495k nuclei in 291 regions | 20x | Warwick TIA, open |
| CoNIC 2022 | Same 6 types, plus per-patch cell counts | 4,981 patches, 256 px (from Lizard) | 20x | grand-challenge |
| CoNSeP | 7 nucleus types (grouped to 4) | 41 images, 1000 px; ~24k nuclei | 40x | Warwick TIA, open |
| CRCHisto | 4 types: epithelial, inflammatory, fibroblast, other | 100 images; ~30k nuclei | 20x | Warwick TIA, open |
| PanNuke | 5 types; colon is one of 19 tissues | ~7.9k patches; ~189k nuclei (all tissues) | 40x | Warwick TIA, open |

Several Warwick datasets have non-commercial licences (CC BY-NC-SA).

---

## 7. Slide-level data for weak supervision / MIL

| Dataset | What is annotated | Size | Scale | Access |
|---|---|---|---|---|
| TCGA COAD / READ | MSI, consensus molecular subtype, KRAS/BRAF, stage, survival (from linked studies) | ~600 patients; FFPE (DX) and frozen (TS/BS) slides | 20x/40x mixed | GDC portal, open |
| CPTAC-COAD | Proteogenomic and clinical data | ~100 patients | 20x/40x | TCIA, open |
| IMP-CRS 2024 | Non-neoplastic / low-grade / high-grade lesion | 5,333 biopsy and polypectomy WSIs | 40x | INESC TEC; CC BY-NC |
| HunCRC WSIs | Slide category (normal, adenoma, CRC, non-neoplastic) plus region outlines | 200 WSIs (MRXS, 392 GB) | 40x-equivalent | TCIA, open |
| TCGA MSI tiles (Kather) | MSI vs MSS; slide label given to every tile | TCGA CRC tiles | 0.5 µm/px | Zenodo 2530835, open |

TCGA frozen slides (TS/BS) are the main public source for frozen-section experiments.

---

## 8. Where each target class can come from

Datasets in **bold** are the best match for this project.

| Class | Patch label | Pixel / region mask | Slide label |
|---|---|---|---|
| Tumour | **HunCRC**, NCT TUM, STARC-9 TUM | **SemiCOL**, PAIP 2020, DigestPath | TCGA, IMP-CRS (high grade) |
| Dysplasia | **HunCRC LG/HG**, **UniToPatho**, EBHI | EBHI-Seg | **IMP-CRS 2024** (low/high grade) |
| Tumour necrosis | **HunCRC**, STARC-9 NCS, NCT DEB | **SemiCOL** (ulcer/necrosis) | — |
| Inflammation | **HunCRC**, NCT LYM, STARC-9 LYM | **SemiCOL** lymphoid; Lizard/CoNIC nuclei | — |
| Stroma | **STARC-9 FCT**, NCT STR/MUS | **SemiCOL** tumour stroma | — |
| Normal mucosa | **HunCRC**, NCT NORM, STARC-9 NOR, UniToPatho | **SemiCOL** mucosa | IMP-CRS (non-neoplastic) |

HunCRC has no stroma. STARC-9 (open, 40x) is the quickest fix, and SemiCOL adds real boundaries.

---

## 9. Suggested order for this project

1. **Now (open data, no agreement)**
   - HunCRC zoom_1 (40x): tumour, dysplasia, necrosis, inflammation, normal
   - STARC-9 or NCT-CRC-HE-100K for stroma
   - Tile classifier on cached Virchow2 features
2. **Next (needs registration or a request)**
   - SemiCOL pixel masks for stroma and real boundaries
   - UniToPatho / EBHI-Seg for dysplasia grade
   - Segmentation decoder trained on mixed labels
3. **Then (slide-level models)**
   - IMP-CRS 2024 for lesion grade per slide
   - TCGA COAD/READ for MSI and subtype; its frozen slides for frozen-section tests
   - Attention-based MIL on the same cached embeddings

---

## 10. Check four things before combining any two datasets

1. **Scale:** resample everything to one µm/px (0.25 for ViSPACE). Never mix 10x and 40x tiles without resampling.
2. **Label definitions:** map every source class to your classes; anything that doesn't fit becomes 255 (ignore).
3. **Stain, scanner, preparation:** use raw (non-normalised) versions plus stain augmentation. All datasets above are FFPE except the TCGA TS/BS frozen slides.
4. **Licence and splits:** several datasets are non-commercial or need agreements (SemiCOL, PAIP, MHIST). Split by patient within every dataset.

---

## 11. Sources

- NCT-CRC-HE-100K: https://zenodo.org/records/1214456
- Kather 2016: https://zenodo.org/records/53169
- STARC-9: https://arxiv.org/abs/2511.00383
- HunCRC: figshare collection 5927795; TCIA "Hungarian-Colorectal-Screening" (https://www.cancerimagingarchive.net/collection/hungarian-colorectal-screening/)
- UniToPatho: https://github.com/EIDOSLAB/UNITOPATHO (IEEE DataPort)
- MHIST: https://bmirds.github.io/MHIST
- Chaoyang: Zhu et al., "Hard sample aware noise robust learning" (arXiv 2112.03694)
- EBHI / EBHI-Seg: https://figshare.com/articles/dataset/EBHI-SEG/21540159
- SemiCOL: https://www.semicol.org/data/
- PAIP 2020: https://paip2020.grand-challenge.org
- DigestPath 2019: https://digestpath2019.grand-challenge.org
- GlaS, CRAG, CoNSeP, CRCHisto, PanNuke, Lizard: https://warwick.ac.uk/fac/cross_fac/tia/data/
- CoNIC 2022: https://conic-challenge.grand-challenge.org
- TCGA COAD/READ: https://portal.gdc.cancer.gov
- CPTAC-COAD: https://www.cancerimagingarchive.net
- IMP-CRS 2024: https://rdm.inesctec.pt/dataset/nis-2023-008
- TCGA MSI tiles: https://zenodo.org/records/2530835
