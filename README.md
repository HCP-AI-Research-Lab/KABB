
# KABB: Knowledge-Aware Bayesian Bandits for Multi-Agent Coordination

> Official code for the ICML 2025 paper [“KABB: Knowledge-Aware Bayesian Bandits for Dynamic Expert Coordination in Multi-Agent Systems”](https://arxiv.org/abs/2502.07350).

\[ [English](README.md) | [中文](README_zh.md) \]

![KABB Logo](https://img.shields.io/badge/ICML-2025-blue) ![Python](https://img.shields.io/badge/Python-3.9%2B-green)

---

## Introduction

KABB (Knowledge-Aware Bayesian Bandits) is a dynamic expert coordination framework for multi-agent systems, featuring:

1. **3D Knowledge Distance Model** — Semantic matching between experts and tasks.
2. **Dual Adaptation Mechanism** — Continuous optimization of expert representation and selection.
3. **Knowledge-Aware Thompson Sampling** — Efficient expert selection in Bayesian MAB with knowledge distance.

See the [paper](https://arxiv.org/abs/2502.07350) for theoretical details and full experiments.

---

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your_org/KABB.git
   cd KABB
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and fill in your API keys
   ```

---

## Quick Start

Run a sample math task with KABB:
```bash
python scripts/run_kabb.py \
    --config configs/config_math_template.yaml \
    --question "What is the value of (7/8)^3 * (7/8)^-3?"
```

---

## Configuration

`configs/config_math_template.yaml` provides a minimal working config, including:
- **system_prompts**: System prompts for different scenarios
- **domain_inference_settings**: Domain priors, samples, and key symbols
- **experts_pool**: List of experts (model, temperature, max tokens, etc.)
- **llm**: API key placeholder (use environment variable)

To extend to other domains:
1. Add a new domain entry with `prior` and `typical_samples`
2. Prepare a set of expert models for the domain
3. Specify the new config in the script

---

## Reproducibility

Reproduce main results with:
```bash
python scripts/run_kabb.py --config configs/config_math_template.yaml --question "..."
```
For full benchmarks, see the `run_*.py` scripts and refer to the paper appendix.

---

## Citation

If you use this project, please cite:
```bibtex
@misc{zhang2025kabbknowledgeawarebayesianbandits,
      title={KABB: Knowledge-Aware Bayesian Bandits for Dynamic Expert Coordination in Multi-Agent Systems}, 
      author={Jusheng Zhang and Zimeng Huang and Yijia Fan and Ningyuan Liu and Mingyan Li and Zhuojie Yang and Jiawei Yao and Jian Wang and Keze Wang},
      year={2025},
      eprint={2502.07350},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2502.07350}, 
}
```

---

## Contributing
- Fork and create a new branch
- Ensure tests pass (`pytest`)
- Follow PEP 8 and use `black` for formatting
- Add tests and docs for new features
- Link related issues in PRs

---

## License

MIT. See [LICENSE](LICENSE).

