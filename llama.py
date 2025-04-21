import os
import psutil
import time
import pandas as pd
import nltk
from llama_cpp import Llama

llama_3_2_1B = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
summllama_3_2_3B = "SummLlama3.2-3B-Q4_K_M.gguf"
hackerllama_3_1_8B = "Hacker-News-Comments-Summarization-Llama-3.1-8B-Instruct.i1-Q4_K_M.gguf"

# CPU és memória mérés előkészítése
process = psutil.Process(os.getpid())
start_time = time.time()

llm = Llama(
      model_path=f"./llm_models/{hackerllama_3_1_8B}",
      n_gpu_layers=-1, 
      n_ctx=4096
)

# Előző memória használat
initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

with open("./Onlab_LLM_2025/prompts/system_prompt_hun.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()

print(system_prompt)
with open("./Onlab_LLM_2025/prompts/user_prompt_whisper_hun.txt", "r", encoding="utf-8") as f:
    user_prompt = f.read().strip()
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

# Modell futtatása
output = llm.create_chat_completion(
    messages=messages,
    max_tokens=2000
)

response_text = output["choices"][0]["message"]["content"].strip()

# Végső memória használat
final_memory = process.memory_info().rss / (1024 * 1024)  # MB

# Tokenek számítása
tokens = nltk.word_tokenize(response_text)
token_count = len(tokens)

output_dir = "Onlab_LLM_2025/outputs"
os.makedirs(output_dir, exist_ok=True)

# Kimenet fájlba írása
output_file_path = os.path.join(output_dir, f"kimenet_whisper_{hackerllama_3_1_8B}.txt")
with open(output_file_path, "w", encoding="utf-8") as f:
    f.write(response_text)

# Idő mérés
end_time = time.time()
elapsed_time = end_time - start_time  # Másodpercekben

# CPU és memória használat kiírása
cpu_usage = psutil.cpu_percent(interval=1)  # CPU használat (1 másodpercet várunk)

# CSV fájl elérési útja
performance_csv_path = os.path.join(output_dir, f"{hackerllama_3_1_8B}_performance_metrics.csv")

# Az adatokat DataFrame-be gyűjtjük
data = {
    'Model': [hackerllama_3_1_8B],
    'Subject': ["whisper"],
    'Generation Time (seconds)': [f"{elapsed_time:.2f}"],
    'CPU Usage (%)': [f"{cpu_usage}"],
    'Memory Usage (MB)': [f"{final_memory - initial_memory:.2f}"],
    'Token Count': [token_count]
}

# Ha a CSV fájl létezik, akkor hozzáadjuk az új adatokat, ha nem, akkor létrehozzuk
if os.path.exists(performance_csv_path):
    df = pd.read_csv(performance_csv_path)
    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
else:
    df = pd.DataFrame(data)

# CSV fájl mentése
df.to_csv(performance_csv_path, index=False)