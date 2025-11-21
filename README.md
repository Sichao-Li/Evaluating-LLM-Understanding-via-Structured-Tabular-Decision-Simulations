# 📘 LLM-Faithfulness: Structured Tabular Decision Simulations (STaDS)

This repository provides **reproducible code and instructions** for evaluating whether Large Language Models (LLMs) understand tabular decision tasks.

The project implements the **STaDS protocol**, which jointly evaluates LLM's understanding at a global level:

- **Predictive Competence** (classification accuracy across instances)  
- **Decision Faithfulness** (feature-importance agreement between behavioral- and claimed-attribution)  

## 📂 Project Structure

```text
├── config/                 # Configuration files
├── data/                   # Dataset CSVs and metadata
│   ├── iris/
│   ├── adult/
│   └── ...
├── src/
│   ├── models.py           # Unified interface for HF, OpenAI, and Gemini models
│   ├── data_loader.py      # Data masking and stratification logic
│   ├── parser.py           # LLM-as-a-Judge output extraction
│   ├── prompts.py          # Jinja2 instruction templates
│   └── metrics.py          # Exact implementation of STaDS metrics (Penalized Accuracy)
├── scripts/
│   ├── run_prediction.py   # Task 1 (Prediction) & Task 3 (Leave-Attribute-Out)
│   └── run_attribution.py  # Task 2 (Self-Attribution/Feature Ranking)
└── requirements.txt
```


# 🔧 Installation

### 1. Clone and Install Dependencies
```
git clone github-repo
cd stads
pip install -r requirements.txt
```

### 2. Set Environment Variable
```
HF_TOKEN=hf_...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

---

# 📊 Datasets

This repository does **not** redistribute datasets.

To reproduce experiments, download the datasets from their original sources (UCI, OpenML, etc.) and place the preprocessed CSVs under:

```
data/<dataset_name>/
    <dataset_name>.ymal
    train.csv
    test.csv
```
---

# 🚀 Running Experiments

## Task 1. Predictive Competence

Example:

```bash
# Run Llama-3.1-8B on the Iris dataset
python scripts/run_prediction.py \
  --dataset iris \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --quantization 4bit \
  --task standard \
  --parse
```

This will:

1. Mask the `class` labels as `class=?`  
2. --parse uses a lightweight LLM (e.g., GPT-4o-mini) to clean the output into a list of integers.
3. --task standard provides standard table prediction
4. Save JSON output

## Task 2: Decision Faithfulness (Self-Attribution)
This task asks the model to rank features by importance without making predictions.

```bash
python scripts/run_attribution.py \
  --dataset iris \
  --model meta-llama/Llama-3.1-8B-Instruct
```
---

## 3. Leave-Attribute-Out (LAO) Attribution
Automates the removal of features one by one to measure the impact on prediction accuracy. This serves as a behavioral feature importance to compare against Task 2.

```bash
python scripts/run_prediction.py \
  --dataset adult \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --task lao \
  --parse
```
---

# 🎯 Metrics
The evaluation uses strict metrics defined in the STaDS protocol:
- Length F1: Harmonic mean of length precision and recall.
- Unknown Label Rate: Percentage of predicted classes that do not exist in the ground truth schema.
- Penalized Accuracy: max(0, Accuracy - α*(1 - Length_F1) - β*(Unknown_Rate))
- Leave-Any-Out (LAO) Attribution: Performance drop from the baseline
- Self-Attribution Recall: Percentage of self-reported important features covering the ground-truth feature set.
- Self-Faith: Agreement between the model’s self-claimed attribution ranking and its actual reliance on features.
- LAO Magnitude: The dispersion of the model’s behavioral reliance across features.
---

# 📄 Citation
Please cite the paper if you use this code or build on the idea. 
```bash
@article{li2025evaluating,
  title={Evaluating LLM Understanding via Structured Tabular Decision Simulations},
  author={Li, Sichao and Xu, Xinyue and Li, Xiaomeng},
  journal={arXiv preprint arXiv:2511.10667},
  year={2025}
}
```