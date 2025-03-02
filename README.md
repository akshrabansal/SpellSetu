# SpellSetu

SpellSetu is a robust bilingual text correction system designed for both English and Hindi. It identifies and fixes spelling errors and improper word usage using a hybrid approach. First, it generates correction candidates via dictionary lookups employing Jaro-Winkler similarity metrics. Then, it re-ranks these candidates based on context using the pre‑trained XLM‑Roberta model. This combination ensures accurate and context-aware corrections for multilingual text.

---

## Problem Statement

SpellSetu addresses the challenge of correcting spelling mistakes and inappropriate word usage in bilingual contexts (English and Hindi). Many existing solutions focus on only one language or do not incorporate context into their corrections, leading to suboptimal results.

---

## Key Features

- **Bilingual Correction:** Supports both English and Hindi.
- **Dictionary-based Corrections:** Uses SCOWL for English and a custom Hindi wordlist.
- **Contextual Re-ranking:** Leverages XLM‑Roberta to select the best correction based on context.
- **Synthetic Data Evaluation:** Generates synthetic error–correct pairs for performance evaluation.
- **Optional Fine-Tuning:** Allows the model to be fine‑tuned on your Hindi wordlist for enhanced performance.

---

## How SpellSetu Works

1. **Dictionary Loading:**  
2. **Error Detection and Correction:**  
3. **Optional Fine-Tuning:**     
4. **Interactive Evaluation:**

---

## Technology Stack

- **Programming Language:** Python
- **Framework:** Streamlit for the interactive UI
- **Libraries and Tools:**
  - [PyTorch](https://pytorch.org/) for deep learning.
  - [Transformers](https://huggingface.co/transformers/) for pre‑trained language models.
  - [TextDistance](https://pypi.org/project/textdistance/) for similarity calculations.
  - [safetensors](https://github.com/huggingface/safetensors) for model checkpointing.
- **Data:** Custom Hindi wordlist and SCOWL-based English wordlist.

---

### Running the App

1. **Install Dependencies**

   Make sure you have Python installed (preferably Python 3.7 or above). Then, install the required packages by running:
   
   ```bash
   pip install streamlit torch transformers textdistance numpy accelerate safetensors
