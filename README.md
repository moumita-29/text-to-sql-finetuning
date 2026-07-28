# Text-to-SQL Generator

Convert plain English questions into SQL queries using a fine-tuned code language model.

## Live Demo
[Try it on HuggingFace Spaces](https://huggingface.co/spaces/moumita-29/codegen-sql-qlora-demo)

## Model
[moumita-29/codegen-sql-qlora on HuggingFace](https://huggingface.co/moumita-29/codegen-sql-qlora)

---

## Results

| Metric | Value |
|--------|-------|
| BLEU Score | 0.45 |
| Training Examples | 20,000 |
| Training Time | ~25 min on free Colab T4 GPU |
| Base Model | Salesforce/codegen-350M-mono |

---

## Example Outputs

| Question | Schema | Generated SQL |
|----------|--------|---------------|
| How many employees earn more than 50000? | employees (id, name, salary, department) | `SELECT COUNT(*) FROM employees WHERE salary > 50000;` |
| List all students who scored above 90 in Math | scores (student_name, subject, score) | `SELECT student_name FROM scores WHERE subject = "Math" AND score > 90;` |
| How many customers are from New York? | customers (id, name, city, email) | `SELECT COUNT(*) FROM customers WHERE city = "New York";` |

---

## Approach

**Base model:** `Salesforce/codegen-350M-mono`
A code-specialized model chosen over general LLMs (like Phi-2) to reduce SQL hallucination. Its smaller size (350M vs 2.7B) allowed training on 20k examples without quantization, resulting in more stable outputs.

**Fine-tuning method:** LoRA (Low-Rank Adaptation)
- Rank `r=16`, alpha=32
- `target_modules="all-linear"`
- Reduces trainable parameters by ~100x vs full fine-tuning
- Entire training fits on a free Colab T4 GPU (15GB VRAM)

**Dataset:** [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)
78k English question → SQL pairs. Trained on 20k examples with a 90/10 train/test split.

**Inference post-processing:**
The model occasionally appends unnecessary SQL clauses. A post-processing step removes noise keywords (INTERSECT, HAVING, OFFSET, GROUP BY) to produce clean, minimal queries.

---

## Repository Structure

```
├── SQL_generator_proj.ipynb   # Full training pipeline
├── app.py                     # Gradio demo (deployed on HuggingFace Spaces)
├── requirements.txt           # Dependencies
└── README.md
```

---

## Run It Yourself

```bash
pip install transformers peft torch gradio
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, re

tokenizer = AutoTokenizer.from_pretrained("moumita-29/codegen-sql-qlora")
base = AutoModelForCausalLM.from_pretrained(
    "Salesforce/codegen-350M-mono",
    torch_dtype=torch.float32
)
model = PeftModel.from_pretrained(base, "moumita-29/codegen-sql-qlora")
model.eval()

def generate_sql(question, schema):
    prompt = f"""### Task: Convert the question to SQL.

### Schema:
{schema}

### Question:
{question}

### SQL:
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql = full.split("### SQL:")[-1].split("###")[0]
    return " ".join(sql.split()).strip() + ";"

print(generate_sql(
    question="How many employees earn more than 50000?",
    schema="CREATE TABLE employees (id INT, name TEXT, salary INT)"
))
# Output: SELECT COUNT(*) FROM employees WHERE salary > 50000;
```

---

## What I Would Improve With More Compute

- Train on the full 78k dataset for better coverage of complex JOIN queries
- Add execution accuracy evaluation (does the SQL actually run and return correct results?)
- Experiment with larger models like CodeLlama-7B
- Add few-shot examples in the prompt for harder multi-table queries

---

## Tech Stack

Python · PyTorch · HuggingFace Transformers · PEFT · TRL · Gradio · Google Colab
