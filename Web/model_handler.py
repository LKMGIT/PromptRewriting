import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 1. Load and preprocess data
def clean_text(text):
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_raw_data(data, min_len=5, max_len=300):
    seen = set()
    cleaned = []
    for item in data:
        inp = clean_text(item["input"])
        tgt = clean_text(item["label"])
        if min_len <= len(inp) <= max_len and inp not in seen:
            cleaned.append({"input": inp, "label": tgt})
            seen.add(inp)
    return cleaned

with open("prompt_dataset_input_label22.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)
cleaned_data = clean_raw_data(raw_data)
good_prompts = [item["label"] for item in cleaned_data]

# 2. Embedding and Indexing
embed_model = SentenceTransformer("jhgan/ko-sbert-sts")
vectors = embed_model.encode(good_prompts, convert_to_numpy=True)
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# 3. Load T5 Model
tokenizer = T5Tokenizer.from_pretrained("KETI-AIR/ke-t5-base")
tokenizer.pad_token = tokenizer.eos_token
model = T5ForConditionalGeneration.from_pretrained("./t5_finetuned_result")

# 4. Retrieval Function
def retrieve_similar_prompts(user_input, top_k=3):
    query_vector = embed_model.encode([user_input])
    _, indices = index.search(np.array(query_vector), top_k)
    return [good_prompts[i] for i in indices[0]]

# 5. Clean Output
def clean_output(text):
    text = text.replace("<pad>", "").replace("</s>", "")
    text = re.sub(r"\b(\w+)(\s+\1)+\b", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    sentences = re.split(r'(?<=[.?!])\s+|\n+', text.strip())
    return sentences[0].strip() if sentences else text.strip()

# 6. Main Inference Function
def rag_refine_prompt(bad_prompt, max_new_tokens=64):
    examples = retrieve_similar_prompts(bad_prompt)
    context = "\n".join([f"예시{i+1}: {ex}" for i, ex in enumerate(examples)])
    prompt = (
        f"사용자 입력: {bad_prompt}\n\n"
        f"{context}\n\n"
        f"위의 예시를 참고해서 프롬프트를 더 구체적이고 명확하게 바꿔주세요.\n출력:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_output(result)
