import pandas as pd

def save_metrics_to_file(reference_path, hypothesis_path, output_file_path, model_name):
    try:
        # Szövegek betöltése
        with open(reference_path, 'r', encoding='utf-8') as f:
            reference = f.read().strip()
        
        with open(hypothesis_path, 'r', encoding='utf-8') as f:
            hypothesis = f.read().strip()
        
        # Tokenizálás
        reference_tokens = nltk.word_tokenize(reference)
        hypothesis_tokens = nltk.word_tokenize(hypothesis)
        
        # BLEU kiszámítása
        bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens)
        
        # METEOR kiszámítása
        meteor = meteor_score([reference_tokens], hypothesis_tokens)
        
        # ROUGE kiszámítása
        rouge = Rouge()
        rouge_scores = rouge.get_scores(hypothesis, reference)[0]
        
        # Eredmények összeállítása
        results = {
            'Model': model_name,
            'Subject': reference_path.split('_')[4].split('.')[0] ,
            'BLEU': bleu_score,
            'METEOR': meteor,
            'ROUGE-1 Precision': rouge_scores['rouge-1']['p'],
            'ROUGE-1 Recall': rouge_scores['rouge-1']['r'],
            'ROUGE-1 F1': rouge_scores['rouge-1']['f'],
            'ROUGE-2 Precision': rouge_scores['rouge-2']['p'],
            'ROUGE-2 Recall': rouge_scores['rouge-2']['r'],
            'ROUGE-2 F1': rouge_scores['rouge-2']['f'],
            'ROUGE-L Precision': rouge_scores['rouge-l']['p'],
            'ROUGE-L Recall': rouge_scores['rouge-l']['r'],
            'ROUGE-L F1': rouge_scores['rouge-l']['f'],
        }
        
        # Eredmények mentése CSV fájlba
        df_results = pd.DataFrame([results])
        
        # Ha létezik már a fájl, akkor új sorokat adunk hozzá
        if os.path.exists(output_file_path):
            df_results.to_csv(output_file_path, mode='a', header=False, index=False)
        else:
            df_results.to_csv(output_file_path, mode='w', header=True, index=False)
        
        print(f"Az eredmények sikeresen mentve a(z) {output_file_path} fájlba.")
        return True
    
    except Exception as e:
        print(f"Hiba történt az eredmények mentése közben: {e}")
        return False

if __name__ == "__main__":
    import os
    import nltk
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge import Rouge
    
    # Ha még nem töltötted le az NLTK csomagokat, kommentezd ki az alábbi sorokat
    # nltk.download('punkt')
    # nltk.download('wordnet')
    
    # Útvonalak beállítása
    reference_path = "./Onlab_LLM_2025/references/referencia_hangfajl_whisper_leirat.txt"
    hypothesis_path = "./Onlab_LLM_2025/outputs/kimenet_whisper_Hacker-News-Comments-Summarization-Llama-3.1-8B-Instruct.i1-Q4_K_M.gguf.txt"
    output_file_path = "./Onlab_LLM_2025/results/Hacker-News-Comments-Summarization-Llama-3.1-8B-Instruct.i1-Q4_K_M.gguf_metrics.csv"
    
    # Eredmények mentése
    save_metrics_to_file(reference_path, hypothesis_path, output_file_path, "Hacker-News-Comments-Summarization-Llama-3.1-8B-Instruct.i1-Q4_K_M")