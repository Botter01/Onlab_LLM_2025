# Onlab_LLM_2025

Ez a projekt a `llama-cpp-python` könyvtárat használja nagy nyelvi modellek (LLM-ek) futtatására lokálisan Pythonból.

## Szükséges modellek

A következő `.gguf` formátumú modellek szükségesek:

- **Hacker-News-Comments-Summarization-Llama-3.1-8B-Instruct.i1-Q4_K_M.gguf**
- **Llama-3.2-1B-Instruct-Q4_K_M.gguf**
- **meta-llama-3-8b-instruct.Q4_K_M.gguf** *(kifejezetten a G-eval tesztekhez használva)*
- **SummLlama3.2-3B-Q4_K_M.gguf**

> **Fontos:** Ezeket a modelleket külön kell letölteni, mivel nem tartalmazza őket a repó.

## Követelmények

- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) könyvtár telepítése szükséges:
  
  ```bash
  pip install llama-cpp-python
