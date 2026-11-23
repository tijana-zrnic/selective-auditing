# Reliable AI-Assisted Decisions via Selective Auditing
By Isaac Gibbs, Elise L. Han, Ryan J. Tibshirani, Tijana Zrnic

> **Code for the paper "[Reliable AI-Assisted Decisions via Selective Auditing](https://arxiv.org/abs/2610.31662)."**

An AI system outputs a sequence of decisions, each with a confidence score. Selective auditing decides online which decisions to send to an expert auditor, with the goal of auditing as rarely as possible while ensuring that at most an $\alpha$ fraction of AI errors escape review. This repository contains the three methods from the paper and reproduces all of its figures.

## Citation
```bibtex
@article{selectiveauditing2026,
  title   = {Reliable AI-Assisted Decisions via Selective Auditing},
  author  = {Isaac Gibbs and Elise L. Han and Ryan J. Tibshirani and Tijana Zrnic},
  journal = {arXiv preprint arXiv:2610.31662},
  year    = {2026},
}
```

## Reproducing the figures
```bash
pip install -r requirements.txt
python plot.py
```
`plot.py` reads the sweep results in `results/` and writes every figure in the paper to `figures/`:

| file | figure |
|---|---|
| `figure1_safety.pdf`, `figure2_annotation.pdf`, `legend.pdf` | Figures 1 and 2 (shared legend) |
| `figure3_labeling.pdf` | Figure 3 |
| `figure4_safety_uncapped.pdf`, `figure5_annotation_uncapped.pdf`, `legend_uncapped.pdf` | Figures 4 and 5 (shared legend) |
| `figure6_labeling_uncapped.pdf` | Figure 6 |
| `figure7_annotation_cutoff{0.7,0.6,0.5}.pdf` | Figure 7 |

To recompute the results from the data:
```bash
python sweep.py                  # all streams
python sweep.py bias imagenetv2  # a subset
```
The sweep runs in parallel on all available cores and takes a few minutes on a laptop.

## Methods
`algorithms.py` implements
* `additive_thresholding` — additive thresholding, equations (1)–(2);
* `additive_thresholding_calibrated` — additive thresholding with confidence calibration, Algorithm 1;
* `optimally_efficient` — optimally efficient selective auditing, Algorithm 2.

Each takes a stream of confidences and error indicators and returns the realized error rate and the fraction of audit effort saved:
```python
from algorithms import additive_thresholding_calibrated
from data import load_safety

conf, err = load_safety("sorrybench")
additive_thresholding_calibrated(conf, err, alpha=0.1, seed=0)
```

## Data
* `data/safety/` — SORRY-Bench and StrongREJECT. `confidence` is the fraction of $K=8$ GPT-5.4 judgments that the response is safe, and `err` is the human unsafe label.
* `data/ucsb/` — qualitative coding of the UCSB interview corpus, with Qwen2.5-14B or GPT-4o-mini as the annotator and GPT-5.2 as the auditor. `confidence` is the annotator's verbalized confidence and `quality_score` is the auditor's grade; an error is `quality_score <= cutoff`.
* `data/pac/` — political bias, stance, ImageNet, and ImageNet-V2, from [PAC Labels](https://github.com/tijana-zrnic/pac-labels). An error is `Y != Yhat`.

Each stream is held fixed across trials. The labeling and safety streams are shuffled once with a fixed seed. Confidences on a coarse grid (bias, stance, safety, interview annotation) are broken up by a small fixed uniform jitter.

## Repository structure
```
selective-auditing/
├── algorithms.py   # Selective auditing methods and the oracle baseline
├── data.py         # Loads each data stream
├── sweep.py        # Runs all methods over α and trials, writes results/
├── plot.py         # Paper figures from results/
├── data/
├── results/        # Precomputed sweep results
└── figures/        # Paper figures
```

## License
This project is released under the MIT License – see the [LICENSE](LICENSE) file for details.
