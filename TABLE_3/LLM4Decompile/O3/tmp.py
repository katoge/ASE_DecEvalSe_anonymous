import json
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu as nltk_bleu, SmoothingFunction
from collections import Counter
from nltk.util import ngrams

import code_bert_score
from codebleu import calc_codebleu
from crystalbleu import corpus_bleu as crystal_bleu

nltk.download("punkt", quiet=True)

TRIVIAL_NGRAMS_PATH = "c_trivial_ngrams.json"
NGRAM_SAMPLE_SIZE = 5000
NGRAM_K = 100
INPUT_FILE = "function_logs.jsonl"
EDIT_DISTANCES_FILE = "edit_distances.txt"
OUTPUT_FILE = "function_metrics.txt"

smoother = SmoothingFunction().method1

def remove_name_conflict(code: str) -> str:
    return code.replace('_name_conflict', '')

def build_trivial_ngrams(sample_size=NGRAM_SAMPLE_SIZE, k=NGRAM_K):
    from datasets import load_dataset
    print(f"Extracting top {k} frequent n-grams from {sample_size} C files...")
    dataset = load_dataset("codeparrot/github-code", split="train", languages=["C"], streaming=True, trust_remote_code=True)

    tokenized_corpus = []
    for i, example in enumerate(dataset):
        if i >= sample_size:
            break
        code = example["code"]
        tokens = word_tokenize(code)
        tokenized_corpus.extend(tokens)

    all_ngrams = []
    for n in range(1, 5):
        all_ngrams.extend(ngrams(tokenized_corpus, n))

    freq = Counter(all_ngrams)
    top_k = dict(freq.most_common(k))

    with open(TRIVIAL_NGRAMS_PATH, "w") as f:
        json.dump({str(k): v for k, v in top_k.items()}, f)

    return top_k

def load_or_build_trivial_ngrams():
    if os.path.exists(TRIVIAL_NGRAMS_PATH):
        with open(TRIVIAL_NGRAMS_PATH, "r") as f:
            raw = json.load(f)
        return {eval(k): v for k, v in raw.items()}
    else:
        return build_trivial_ngrams()

def compute_metrics(reference: str, prediction: str, trivial_ngrams):
    ref_tokens = word_tokenize(reference)
    pred_tokens = word_tokenize(prediction)

    cb_score = code_bert_score.score(cands=[prediction], refs=[reference], lang="c")[2][0].item()
    codebleu_score = calc_codebleu([reference], [prediction], lang="c",
                                   weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)["codebleu"]
    crystalbleu_score = crystal_bleu([[ref_tokens]], [pred_tokens],
                                     ignoring=trivial_ngrams,
                                     smoothing_function=smoother)
    nltk_corpusbleu = nltk_bleu([[ref_tokens]], [pred_tokens], smoothing_function=smoother)

    return cb_score, codebleu_score, crystalbleu_score, nltk_corpusbleu

def main():
    trivial_ngrams = load_or_build_trivial_ngrams()

    with open(INPUT_FILE, "r") as infile:
        json_lines = [json.loads(line) for line in infile.readlines()]

    with open(EDIT_DISTANCES_FILE, "r") as f:
        edit_distances = [float(line.split()[2]) for line in f.readlines()]

    if len(json_lines) != len(edit_distances):
        raise ValueError("Mismatch between JSONL entries and edit distance lines")

    results = []
    for idx, (entry, lev_dist) in enumerate(zip(json_lines, edit_distances)):
        reference = entry.get("function", "")
        prediction = remove_name_conflict(entry.get("function_prediction", ""))

        try:
            metrics = compute_metrics(reference, prediction, trivial_ngrams)
        except Exception as e:
            print(f"Error on line {idx + 1}: {e}")
            metrics = (0.0, 0.0, 0.0, 0.0)

        results.append((lev_dist, *metrics))

    # Write results and compute averages
    metric_sums = [0.0] * 5
    with open(OUTPUT_FILE, "w") as out:
        for row in results:
            metric_sums = [a + b for a, b in zip(metric_sums, row)]
            out.write(" ".join(f"{val:.6f}" for val in row) + "\n")

        averages = [val / len(results) for val in metric_sums]
        out.write("AVERAGE " + " ".join(f"{val:.6f}" for val in averages) + "\n")

    print(f"Evaluation complete. Results written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
