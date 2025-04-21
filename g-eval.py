from llama_cpp import Llama
import pandas as pd
import re, os

# Tesztanyagok beolvasása
with open("./Onlab_LLM_2025/hangfajl_rendes_leirat.txt", "r", encoding="utf-8") as f:
    reference_text = f.read()

with open("./Onlab_LLM_2025/outputs/kimenet_hangfajl_rendes_leirat_Hacker-News-Comments-Summarization-Llama-3.1-8B-Instruct.i1-Q4_K_M.gguf.txt", "r", encoding="utf-8") as f:
    hypothesis_text = f.read()

# Llama 3 modell betöltése a helyes útvonallal
llm = Llama(
    model_path="./llm_models/meta-llama-3-8b-instruct.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1
)

# G-EVAL prompt készítése Llama 3 formátumban
def create_geval_prompt(reference, hypothesis, criteria):
    prompt = f"""<|system|>
Te egy szakértő szövegértékelő asszisztens vagy. A feladatod az alábbi összegzés értékelése az eredeti szöveg alapján.
</|system|>

<|user|>
# Eredeti szöveg
{reference}

# Összegzés
{hypothesis}

# Értékelési kritériumok
Értékeld a fenti összegzést az alábbi szempontok szerint, 1-től 10-ig terjedő skálán, úgy hogy tized pontokat is adhatsz:
- {criteria}

Minden szempont esetén add meg a pontszámot (1.0-10.0), és részletesen indokold döntésedet példákkal az eredeti és az összefoglaló szövegből. Ha pedig teljesen más témáról van szó az összegzésben akkor kevés pontot adj.
Végül számold ki az átlagpontszámot is ilyen formában: (1.0-10.0). De ne adj overall score-t.
</|user|>

<|assistant|>"""
    return prompt

# Értékelés futtatása
criteria = "pontosság, koherencia, tömörség"
output_file_path = "./Onlab_LLM_2025/results/Hacker-News-Comments-Summarization-Llama-3.1-8B-Instruct.i1-Q4_K_M.gguf_g_eval_metrics.csv"
prompt = create_geval_prompt(reference_text, hypothesis_text, criteria)

# Generálás
output = llm(
    prompt,
    max_tokens=2048,
    temperature=0.1
)

text = output["choices"][0]["text"]
print(text)


results = {
        "Model": "Hacker-News-Comments-Summarization-Llama-3.1-8B-Instruct.i1-Q4_K_M",
        "Subject": "hangfajl_rendes_leirat",
    }
df_results = pd.DataFrame([results])
        
# Ha létezik már a fájl, akkor új sorokat adunk hozzá
if os.path.exists(output_file_path):
    df_results.to_csv(output_file_path, mode='a', header=False, index=False)
else:
    df_results.to_csv(output_file_path, mode='w', header=True, index=False)