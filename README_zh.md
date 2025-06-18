
# KABB: 面向多智能体系统的知识感知贝叶斯Bandit专家协调（ICML 2025）

> 本仓库为 ICML 2025 论文 [“KABB: Knowledge-Aware Bayesian Bandits for Dynamic Expert Coordination in Multi-Agent Systems”](https://arxiv.org/abs/2502.07350) 官方代码。


![KABB Logo](https://img.shields.io/badge/ICML-2025-blue) ![Python](https://img.shields.io/badge/Python-3.9%2B-green)


---

## 项目简介

KABB（Knowledge-Aware Bayesian Bandits）是一种面向多智能体系统的动态专家协调框架，核心特性：

1. **三维知识距离模型** —— 语义层面刻画专家与任务的匹配度
2. **双重自适应机制** —— 持续优化专家能力表示与选择策略
3. **知识感知Thompson采样** —— 在贝叶斯MAB中高效选择专家

详细理论与实验结果请见[论文原文](https://arxiv.org/abs/2502.07350)。

---

## 安装

1. 克隆仓库：
   ```bash
   git clone https://github.com/your_org/KABB.git
   cd KABB
   ```
2. （可选）创建虚拟环境：
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
4. 配置环境变量：
   ```bash
   cp .env.example .env
   # 编辑 .env 并填写你的API密钥
   ```

---

## 快速开始

运行一个MATH领域任务示例：
```bash
python scripts/run_kabb.py \
    --config configs/config_math_template.yaml \
    --question "What is the value of (7/8)^3 * (7/8)^-3?"
```

---

## 目录结构

---

## 配置说明

`configs/config_math_template.yaml` 提供了最小可运行配置，包括：
- **system_prompts**：系统提示词
- **domain_inference_settings**：领域先验、样例与关键符号
- **experts_pool**：专家列表（模型、温度、最大token等）
- **llm**：API key 占位符（推荐用环境变量）

如需扩展到其它领域：
1. 新增领域条目并设置 `prior` / `typical_samples`
2. 为该领域准备一组专家模型
3. 在脚本中指定新的配置文件

---

## 实验复现

复现主要实验：
```bash
python scripts/run_kabb.py --config configs/config_math_template.yaml --question "..."
```
完整评测见 `run_*.py` 脚本及论文附录。

---

## 引用

如果本项目对你的研究有帮助，请引用：
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

## 贡献指南
- Fork 并新建分支
- 确保通过 `pytest`
- 遵循 PEP 8 / 使用 `black` 格式化
- 新功能请补充测试与文档
- PR 关联相关 Issue

---

## 许可证

MIT，详见 [LICENSE](LICENSE)。

--- 