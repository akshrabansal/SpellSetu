import streamlit as st
import torch, math, random
import textdistance
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import numpy as np
import os, warnings

logging.set_verbosity_error()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=DeprecationWarning)


#######################################
# 1. Load Dictionaries (SCOWL + Indic)
#######################################
@st.cache_data
def load_dictionary(file_path, file_mod_time, version=1, to_lower=True):
    words = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if to_lower:
                w = w.lower()
            if w:
                words.add(w)
    return words

# File paths (update Hindi file name)
english_words_file = r"C:\Users\Think\Desktop\LINGUA\English_corpus.txt"
hindi_words_file = r"C:\Users\Think\Desktop\LINGUA\Hindi_wordlist.txt"

# Get last modified times
english_mod_time = os.path.getmtime(english_words_file)
hindi_mod_time = os.path.getmtime(hindi_words_file)

# Load dictionaries using modification time
english_words = load_dictionary(english_words_file, english_mod_time, version=2, to_lower=True)
hindi_words = load_dictionary(hindi_words_file, hindi_mod_time, version=2, to_lower=False)

#######################################
# 2. Load Trained Model & Tokenizer
#######################################
# Load your trained bilingual model from the saved directory
model_dir = "./final_xlmr_bilingual"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForMaskedLM.from_pretrained(model_dir)
model.eval()  # Set to evaluation mode

#######################################
# 3. (Optional) Training Routine
#######################################
# (Commented out since the model is already trained)
# class WordDataset(torch.utils.data.Dataset):
#     def __init__(self, tokenizer, words, block_size=32):
#         self.examples = []
#         for w in words:
#             encoded = tokenizer(w, truncation=True, max_length=block_size, padding="max_length", return_tensors="pt")
#             encoded = {k: v.squeeze(0) for k, v in encoded.items()}
#             self.examples.append(encoded)
#     def __len__(self):
#         return len(self.examples)
#     def __getitem__(self, idx):
#         return self.examples[idx]
# st.subheader("Model Training on Hindi Wordlist")
# if st.button("Train on Hindi Wordlist"):
#     st.write("Starting training on Hindi wordlist...")
#     dataset = WordDataset(tokenizer, list(hindi_words), block_size=32)
#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
#     training_args = TrainingArguments(
#         output_dir="./hindi_finetuned_model",
#         overwrite_output_dir=True,
#         num_train_epochs=1,
#         per_device_train_batch_size=4,
#         save_strategy="no",
#         logging_steps=50,
#     )
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=dataset,
#     )
#     trainer.train()
#     st.write("Training completed!")
#     model.eval()

#######################################
# 4. Utility Functions for Correction
#######################################
def is_hindi(word):
    hindi_chars = [c for c in word if '\u0900' <= c <= '\u097F']
    return (len(hindi_chars) / len(word)) > 0.5 if len(word) else False

def generate_candidates_jaro_winkler(token, dictionary_set, threshold=0.75, top_n=5):
    results = []
    t_lower = token.lower()
    for w in dictionary_set:
        similarity = textdistance.jaro_winkler.normalized_similarity(t_lower, w.lower())
        if similarity >= threshold:
            results.append((w, similarity))
    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:top_n]]

def dictionary_correction(tokens, eng_dict, hin_dict, threshold=0.75):
    corrected_tokens = []
    for t in tokens:
        if is_hindi(t):
            if t not in hin_dict:
                candidates = generate_candidates_jaro_winkler(t, hin_dict, threshold)
                corrected_tokens.append(candidates[0] if candidates else t)
            else:
                corrected_tokens.append(t)
        else:
            t_lower = t.lower()
            if t_lower not in eng_dict:
                candidates = generate_candidates_jaro_winkler(t, eng_dict, threshold)
                corrected_tokens.append(candidates[0] if candidates else t)
            else:
                corrected_tokens.append(t)
    return corrected_tokens

# --- OLD Synthetic Logic: Increased error probability to 40%
def introduce_errors_en(word):
    if len(word) > 2 and random.random() < 0.4:
        idx = random.randint(0, len(word) - 2)
        word = word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]
    return word

def introduce_errors_hi(word):
    if len(word) > 2 and random.random() < 0.4:
        idx = random.randint(0, len(word) - 2)
        word = word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]
    return word

# --- Modified Synthetic Data Generation:
# For each sample, split the paragraph into sentences and randomly select 'sentence_count' sentences.
def generate_synthetic_data_separate(correct_sentences, sentence_count=6):
    pairs = []
    for correct in correct_sentences:
        # Split paragraph into sentences (using a period as delimiter; adjust as needed)
        sentences = [s.strip() for s in correct.split('.') if s.strip()]
        # Sample up to 'sentence_count' sentences (if available)
        selected = random.sample(sentences, min(sentence_count, len(sentences)))
        correct_sample = ". ".join(selected) + "."
        tokens = correct_sample.split()
        incorrect_tokens = []
        for t in tokens:
            if is_hindi(t):
                t_inc = introduce_errors_hi(t)
            else:
                t_inc = introduce_errors_en(t)
            incorrect_tokens.append(t_inc)
        incorrect = " ".join(incorrect_tokens)
        pairs.append((incorrect, correct_sample))
    return pairs

def score_candidate_in_context(tokens, idx, candidate, model, tokenizer):
    masked_tokens = tokens.copy()
    masked_tokens[idx] = tokenizer.mask_token
    text = " ".join(masked_tokens)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]
    mask_index = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()
    candidate_subwords = tokenizer.tokenize(candidate)
    if not candidate_subwords:
        return -math.inf
    total_score = 0.0
    for sub in candidate_subwords:
        candidate_id = tokenizer.convert_tokens_to_ids(sub)
        if candidate_id is None:
            return -math.inf
        log_probs = torch.log_softmax(logits[mask_index], dim=-1)
        total_score += log_probs[candidate_id].item()
    average_score = total_score / len(candidate_subwords)
    return average_score

def contextual_correction(tokens, eng_dict, hin_dict, model, tokenizer, threshold=0.75, top_n=5):
    corrected = tokens.copy()
    for i, t in enumerate(tokens):
        if is_hindi(t):
            if t not in hin_dict:
                candidates = generate_candidates_jaro_winkler(t, hin_dict, threshold, top_n)
            else:
                continue
        else:
            if t.lower() not in eng_dict:
                candidates = generate_candidates_jaro_winkler(t, eng_dict, threshold, top_n)
            else:
                continue
        if not candidates:
            continue
        best_candidate = None
        best_score = -math.inf
        for c in candidates:
            score = score_candidate_in_context(corrected, i, c, model, tokenizer)
            if score > best_score:
                best_score = score
                best_candidate = c
        if best_candidate is not None:
            corrected[i] = best_candidate
    return corrected

def hybrid_correct_sentence(sentence, eng_dict, hin_dict, model, tokenizer):
    tokens = sentence.split()
    dict_fixed = dictionary_correction(tokens, eng_dict, hin_dict, threshold=0.75)
    final_tokens = contextual_correction(dict_fixed, eng_dict, hin_dict, model, tokenizer, threshold=0.75)
    return " ".join(final_tokens)

def word_level_accuracy(pred, gold):
    pred_tokens = [p.lower() for p in pred.split()]
    gold_tokens = [g.lower() for g in gold.split()]
    matches = sum(1 for p, g in zip(pred_tokens, gold_tokens) if p == g)
    return matches / max(len(gold_tokens), 1)

#######################################
# 5. Corpora for Synthetic Data
#######################################
english_corpus = [
    """Birds are a group of warm-blooded vertebrates characterized by feathers, toothless beaks, and the laying of hard-shelled eggs. They live worldwide and range in size from the tiny bee hummingbird to the large ostrich. Birds are descendants of theropod dinosaurs and are considered the only living dinosaurs. Their evolution is marked by the development of wings and the loss of flight in some species. Many species are social and communicate with visual signals, calls, and songs."""
]

hindi_corpus = [
    """भारत (आधिकारिक नाम: भारत गणराज्य, अंग्रेज़ी: Republic of India) दक्षिण एशिया में स्थित एक विशाल देश है। यह देश अपनी समृद्ध संस्कृति, ऐतिहासिक धरोहर और विविध भाषाओं के लिए जाना जाता है। यहां के लोग विभिन्न धर्मों और परंपराओं का पालन करते हैं और इसका इतिहास अत्यंत गौरवपूर्ण है।"""
]

#######################################
# 6. Streamlit App UI for Correction
#######################################
st.title("SpellSetu")
st.write("This bilingual correction tool identifies and corrects spelling mistakes and inappropriate word usage in English and Hindi.")

# Create two vertical boxes (columns) side by side for interactive correction
col1, col2 = st.columns(2)

with col1:
    st.header("Incorrect Sentence")
    user_sentence = st.text_area("Enter your incorrect sentence here:",
                                 placeholder="Type an incorrect sentence...", height=150)

with col2:
    st.header("Corrected Sentence")
    corrected_placeholder = st.empty()  # Placeholder for corrected output

if user_sentence:
    corrected_sentence = hybrid_correct_sentence(user_sentence, english_words, hindi_words, model, tokenizer)
    corrected_placeholder.text_area("Output", value=corrected_sentence, height=150, disabled=True)
else:
    corrected_placeholder.text_area("Output", value="", height=150, disabled=True)

# Synthetic Data Evaluation Section for English
st.subheader("Synthetic Data Evaluation - English")
if st.button("Run Synthetic Evaluation - English"):
    eng_pairs = generate_synthetic_data_separate(english_corpus, sentence_count=6)
    st.write("### Synthetic English Pairs & Corrections:")
    scores_eng = []
    for incorr, corr in eng_pairs:
        corrected = hybrid_correct_sentence(incorr, english_words, hindi_words, model, tokenizer)
        st.write(f"**Incorrect:** {incorr}")
        st.write(f"**Corrected:** {corrected}")
        st.write(f"**Expected:** {corr}")
        st.write("---")
        scores_eng.append(word_level_accuracy(corrected, corr))
    avg_accuracy_eng = (sum(scores_eng) / len(scores_eng)) * 100
    st.write("### Average Word-Level Accuracy (English):", f"{avg_accuracy_eng:.2f}%")

# Synthetic Data Evaluation Section for Hindi
st.subheader("Synthetic Data Evaluation - Hindi")
if st.button("Run Synthetic Evaluation - Hindi"):
    hin_pairs = generate_synthetic_data_separate(hindi_corpus, sentence_count=6)
    st.write("### Synthetic Hindi Pairs & Corrections:")
    scores_hin = []
    for incorr, corr in hin_pairs:
        corrected = hybrid_correct_sentence(incorr, english_words, hindi_words, model, tokenizer)
        st.write(f"**Incorrect:** {incorr}")
        st.write(f"**Corrected:** {corrected}")
        st.write(f"**Expected:** {corr}")
        st.write("---")
        scores_hin.append(word_level_accuracy(corrected, corr))
    avg_accuracy_hin = (sum(scores_hin) / len(scores_hin)) * 100
    st.write("### Average Word-Level Accuracy (Hindi):", f"{avg_accuracy_hin:.2f}%")
