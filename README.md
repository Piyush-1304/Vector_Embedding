# Hindi Word Embeddings

Word embeddings from scratch using a **custom Hindi News dataset** (80 rows × 5 columns), SentencePiece BPE tokenizer, and PyTorch.

## Dataset — `hindi_dataset.csv`

| Column | Description |
|---|---|
| `id` | Row number |
| `text` | Hindi sentence (Devanagari) |
| `category` | खेल / राजनीति / शिक्षा / स्वास्थ्य / प्रौद्योगिकी / मनोरंजन / व्यापार / विज्ञान |
| `sentiment` | सकारात्मक / नकारात्मक / तटस्थ |
| `label` | 1 = positive, 0 = negative, 2 = neutral |

## Pipeline Steps
1. Generate Hindi dataset CSV
2. Load & explore with pandas
3. Build raw text corpus
4. Hindi tokenization (Devanagari Unicode regex)
5. Build vocabulary with special tokens
6. Encode / Decode sentences
7. Train SentencePiece BPE tokenizer
8. BPE encoding with subword pieces
9. Padded token ID tensor
10. Token Embedding Layer
11. Positional Embedding Layer
12. Final Input Embeddings
13. Sentence-level vectors (mean pooling)
14. Cosine similarity between sentences
15. Full pipeline summary

## Requirements

```bash
pip install -r requirements.txt
```

## Run

Open `hindi_word_embeddings.ipynb` in Jupyter or VS Code and run all cells.
