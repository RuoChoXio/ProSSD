import argparse
import os
import pathlib
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from models import build_model
from dataloader import load_data, load_parallel_data, load_data_pairs
from method import StyleProjector, FeatureExtractor, StyleLibrary
from statistics import calculate_metrics, save_results, get_optimal_threshold

def get_exp_dir(base_output, model_name, k):
    m_name = model_name.strip("/").split("/")[-1]
    exp_name = f"{m_name}_k{k}"
    path = pathlib.Path(base_output) / exp_name
    path.mkdir(parents=True, exist_ok=True)
    return path

def train_probe(args):
    print(f"=== Task: Train Difference Probe (k={args.k}) ===")
    exp_dir = get_exp_dir(args.output_base, args.model_path, args.k)

    data = load_data_pairs(args.train_data, "human_text", "direct_prompt", limit=args.limit)
    human_texts = [p[0] for p in data]
    ai_texts = [p[1] for p in data]

    tokenizer, model = build_model(args.model_path, args.device)

    def get_pooled_embeds(texts, desc):
        vecs = []
        bsz = args.batch_size
        for i in tqdm(range(0, len(texts), bsz), desc=desc, total=max(1, (len(texts) + bsz - 1) // bsz)):
            batch = texts[i:i+bsz]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(args.device)
            with torch.no_grad():
                out = model(**enc)
                mask = enc.attention_mask.unsqueeze(-1)
                mean = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                vecs.append(mean.cpu().numpy())
        return np.concatenate(vecs, axis=0)

    h_emb = get_pooled_embeds(human_texts, desc="Encoding Human texts")
    a_emb = get_pooled_embeds(ai_texts, desc="Encoding AI texts")

    projector = StyleProjector(k=args.k)
    projector.fit(h_emb, a_emb)

    save_path = exp_dir / "projection.npy"
    projector.save(save_path)
    print(f"Probe saved to: {save_path}")


def build_lib(args):
    print(f"=== Task: Build Weighted Style Library ===")
    exp_dir = get_exp_dir(args.output_base, args.model_path, args.k)
    proj_path = exp_dir / "projection.npy"

    if not proj_path.exists():
        raise FileNotFoundError(f"Projection matrix not found at {proj_path}. Run train_probe first.")

    tokenizer, model = build_model(args.model_path, args.device)
    projector = StyleProjector()
    projector.load(str(proj_path))

    extractor = FeatureExtractor(tokenizer, model, projector, args.device)

    data = load_data_pairs(args.train_data, "human_text", "direct_prompt", limit=args.limit)
    human_texts = [p[0] for p in data]
    ai_texts = [p[1] for p in data]

    print("Extracting Human Features...")
    h_slices = extractor.process_texts(human_texts, args.batch_size, args.spacy_workers)

    print("Extracting AI Features...")
    a_slices = extractor.process_texts(ai_texts, args.batch_size, args.spacy_workers)

    lib = StyleLibrary()
    lib.build(h_slices, a_slices)

    print("\n[Threshold] Calculating optimal threshold on TRAINING data...")
    h_scores = [lib.score(s)[2] for s in tqdm(h_slices, desc="Scoring Train Human")]
    a_scores = [lib.score(s)[2] for s in tqdm(a_slices, desc="Scoring Train AI")]

    best_threshold = get_optimal_threshold(h_scores, a_scores)
    print(f"[Threshold] Optimal Threshold found on Train Set: {best_threshold}")

    lib.set_threshold(best_threshold)

    save_path = exp_dir / "style_lib.pkl"
    lib.save(save_path)
    print(f"Library saved to: {save_path}")


def eval_model(args):
    print(f"=== Task: Evaluation ===")
    exp_dir = get_exp_dir(args.output_base, args.model_path, args.k)
    proj_path = exp_dir / "projection.npy"
    lib_path = exp_dir / "style_lib.pkl"

    if not proj_path.exists() or not lib_path.exists():
        raise FileNotFoundError(f"Model files missing in {exp_dir}. Run train & build first.")

    tokenizer, model = build_model(args.model_path, args.device)
    projector = StyleProjector()
    projector.load(str(proj_path))
    lib = StyleLibrary()
    lib.load(str(lib_path))

    extractor = FeatureExtractor(tokenizer, model, projector, args.device)

    data = load_data_pairs(args.test_data, "human_text", "direct_prompt", limit=args.limit)
    human_texts = [p[0] for p in data]
    ai_texts = [p[1] for p in data]

    print(f"Loaded {len(human_texts)} Human texts, {len(ai_texts)} AI texts.")

    print("Scoring Human texts...")
    h_slices_list = extractor.process_texts(human_texts, args.batch_size, args.spacy_workers)
    h_scores = [lib.score(s)[2] for s in tqdm(h_slices_list, desc="Calculating Human scores")]

    print("Scoring AI texts...")
    a_slices_list = extractor.process_texts(ai_texts, args.batch_size, args.spacy_workers)
    a_scores = [lib.score(s)[2] for s in tqdm(a_slices_list, desc="Calculating AI scores")]

    print("Calculating Metrics...")

    config_info = {
        "model_path": args.model_path,
        "test_data": args.test_data,
        "k": args.k,
        "device": args.device,
        "batch_size": args.batch_size
    }

    print(f"Using fixed threshold from training: {lib.threshold}")
    metrics = calculate_metrics(h_scores, a_scores, threshold=lib.threshold)

    base_dir = pathlib.Path(args.output_base).parent / "results"

    save_results(metrics, config_info, base_dir)

    print("-" * 40)
    print(f"Eval Results: {pathlib.Path(args.model_path).name} (k={args.k})")
    print("-" * 40)
    print(f"AUC:             {metrics['roc_auc']}%")
    print(f"Accuracy:        {metrics['accuracy']}%")
    print(f"F1 Score:        {metrics['f1_score']}%")
    print(f"TPR@0.01% FPR:   {metrics['tpr_at_fpr_0_01_percent']}%")
    print(f"Optimal Thresh:  {metrics['threshold']}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="task", required=True)

    # Common Args
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--model-path", default="./roberta-large")
    parent.add_argument("--device", default="cuda:0")
    parent.add_argument("--batch-size", type=int, default=32)
    parent.add_argument("--output-base", default="./experiments")
    parent.add_argument("--k", type=int, default=32)
    parent.add_argument("--limit", type=int, default=None, help="Debugging limit")

    # Task: Train Probe
    p_train = subparsers.add_parser("train_probe", parents=[parent])
    p_train.add_argument("--train-data", required=True)

    # Task: Build Lib
    p_build = subparsers.add_parser("build_lib", parents=[parent])
    p_build.add_argument("--train-data", required=True)
    p_build.add_argument("--spacy-workers", type=int, default=4)

    # Task: Eval
    p_eval = subparsers.add_parser("eval", parents=[parent])
    p_eval.add_argument("--test-data", required=True)
    p_eval.add_argument("--spacy-workers", type=int, default=4)

    args = parser.parse_args()

    if args.task == "train_probe":
        train_probe(args)
    elif args.task == "build_lib":
        build_lib(args)
    elif args.task == "eval":
        eval_model(args)
