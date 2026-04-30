import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

MODEL = "Salesforce/codegen-350M-mono"
ADAPTER = "moumita-29/codegen-sql-qlora"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

def clean_sql(raw_sql):
    noise_keywords = ["INTERSECT", "UNION", "EXCEPT", "HAVING", "OFFSET", "GROUP BY"]
    raw_upper = raw_sql.upper()
    cut_pos = len(raw_sql)
    for keyword in noise_keywords:
        pos = raw_upper.find(keyword)
        if pos != -1 and pos < cut_pos:
            cut_pos = pos
    sql = raw_sql[:cut_pos].strip()
    sql = re.sub(r"[\s,;)]+$", "", sql)
    return sql + ";"

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
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_sql = full.split("### SQL:")[-1]
    raw_sql = raw_sql.split("###")[0]
    raw_sql = " ".join(raw_sql.split()).strip()
    return clean_sql(raw_sql)

demo = gr.Interface(
    fn=generate_sql,
    inputs=[
        gr.Textbox(label="Your Question",
                   placeholder="How many employees earn more than 50000?"),
        gr.Textbox(label="Table Schema",
                   placeholder="CREATE TABLE employees (id INT, name TEXT, salary INT)")
    ],
    outputs=gr.Textbox(label="Generated SQL"),
    title="Text-to-SQL Generator",
    description="Fine-tuned Salesforce/codegen-350M on 20,000 SQL examples using LoRA | BLEU: 0.38",
    examples=[
        ["How many employees earn more than 50000?",
         "CREATE TABLE employees (id INT, name TEXT, salary INT, department TEXT)"],
        ["List all students who scored above 90 in Math",
         "CREATE TABLE scores (student_name TEXT, subject TEXT, score INT)"],
        ["How many customers are from New York?",
         "CREATE TABLE customers (id INT, name TEXT, city TEXT, email TEXT)"],
    ]
)

demo.launch()
