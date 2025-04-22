from llama_cpp import Llama
import pandas as pd
import re, os

# Tesztanyagok beolvasása
with open("./Onlab_LLM_2025/whisper_leirat_2_15_min.txt", "r", encoding="utf-8") as f:
    reference_text = f.read()

with open("./Onlab_LLM_2025/outputs/kimenet_whisper_2_Llama-3.2-1B-Instruct-Q4_K_M.gguf.txt", "r", encoding="utf-8") as f:
    hypothesis_text = f.read()

# Llama 3 modell betöltése a helyes útvonallal
llm = Llama(
    model_path="./llm_models/meta-llama-3-8b-instruct.Q4_K_M.gguf",
    n_ctx=9000,
    n_gpu_layers=-1
)

# G-EVAL prompt készítése Llama 3 formátumban
def create_geval_prompt(reference, hypothesis, criteria):
    prompt = f"""<|system|>
Te egy szakértő szövegértékelő asszisztens vagy. A feladatod az alábbi összegzés értékelése az eredeti szöveg alapján. Nagyon figyelj arra mi szerepel az összegzésben.
</|system|>

<|user|>
# Eredeti szöveg
{reference}

# Összegzés
{hypothesis}

# Értékelési kritériumok
Értékeld a fenti összegzést az alábbi szempontok szerint, 1-től 10-ig terjedő skálán, úgy hogy tized pontokat is adhatsz:
- {criteria}

Minden szempont esetén add meg a pontszámot (1.0-10.0). Ha pedig teljesen más témáról van szó az összegzésben akkor kevés pontot adj. Ha csak ismétlés van akkor is vonj le pontot.
Végül számold ki az átlagpontszámot is ilyen formában: (1.0-10.0) és rövid leírást, hogy miért. Ha nem jelennek meg főszempontok akkor pontozd le az összegzést.
</|user|>

<|assistant|>"""
    return prompt

# Értékelés futtatása
criteria = "pontosság, koherencia, tömörség"
output_file_path = "./Onlab_LLM_2025/results/Llama-3.2-1B-Instruct-Q4_K_M.gguf_g_eval_metrics.csv"
prompt = create_geval_prompt(reference_text, hypothesis_text, criteria)

# Generálás
output = llm(
    prompt,
    max_tokens=10000,
    temperature=0.1
)

text = output["choices"][0]["text"]
print(text)


results = {
        "Model": "Llama-3.2-1B-Instruct-Q4_K_M",
        "Subject": "hangfajl_whisper_leirat_2",
    }
df_results = pd.DataFrame([results])
        
# Ha létezik már a fájl, akkor új sorokat adunk hozzá
if os.path.exists(output_file_path):
    df_results.to_csv(output_file_path, mode='a', header=False, index=False)
else:
    df_results.to_csv(output_file_path, mode='w', header=True, index=False)