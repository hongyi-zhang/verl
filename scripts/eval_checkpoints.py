#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple
import subprocess

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional: use VERL merger if HF weights are not present
try:
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger
except Exception:
    ModelMergerConfig = None
    FSDPModelMerger = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints at given steps on a QA-style parquet dataset."
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="checkpoints/pcsd_demo/pcsd_news",
        help="Root dir containing global_step_*/actor checkpoints.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="100,200,300,400,500",
        help="Comma-separated training steps to evaluate, e.g. '100,200,300'.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/workspace/pcsd_data/news_all.parquet",
        help="Parquet dataset path.",
    )
    parser.add_argument(
        "--question_col",
        type=str,
        default="question",
        help="Column name for question/prompt.",
    )
    parser.add_argument(
        "--answer_col",
        type=str,
        default="answer",
        help="Column name for target/answer.",
    )
    parser.add_argument(
        "--answer_json_col",
        type=str,
        default=None,
        help="If answers are stored in a JSON-like column (e.g., 'extra_info'), specify the column name here.",
    )
    parser.add_argument(
        "--answer_json_key",
        type=str,
        default=None,
        help="When using --answer_json_col, the key inside that JSON/dict which contains the ground-truth answer (e.g., 'answer').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Evaluate first N samples (set -1 for all).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/verl/outputs/eval_checkpoints",
        help="Directory to save evaluation reports.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="max_new_tokens for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (0 implies deterministic greedy).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="top_p for generation.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass through to transformers.from_pretrained",
    )
    parser.add_argument(
        "--save_jsonl",
        action="store_true",
        help="If set, also export predictions to JSONL per step and a combined JSONL.",
    )
    return parser.parse_args()


def whitespace_tokenize(text: str) -> List[str]:
    return text.strip().split()


def lcs_length(a_tokens: List[str], b_tokens: List[str]) -> int:
    # Classic DP for LCS length, O(n*m). For small eval sizes it's fine.
    n, m = len(a_tokens), len(b_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a_tokens[i - 1]
        dpi = dp[i]
        dpim1 = dp[i - 1]
        for j in range(1, m + 1):
            if ai == b_tokens[j - 1]:
                dpi[j] = dpim1[j - 1] + 1
            else:
                dpi[j] = dpim1[j] if dpim1[j] >= dpi[j - 1] else dpi[j - 1]
    return dp[n][m]


def rouge_l_f1(pred: str, ref: str) -> float:
    pred_tok = whitespace_tokenize(pred)
    ref_tok = whitespace_tokenize(ref)
    if not pred_tok or not ref_tok:
        return 0.0
    lcs = lcs_length(pred_tok, ref_tok)
    prec = lcs / max(len(pred_tok), 1)
    rec = lcs / max(len(ref_tok), 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def token_f1(pred: str, ref: str) -> float:
    pred_tok = whitespace_tokenize(pred)
    ref_tok = whitespace_tokenize(ref)
    if not pred_tok and not ref_tok:
        return 1.0
    if not pred_tok or not ref_tok:
        return 0.0
    pred_set = set(pred_tok)
    ref_set = set(ref_tok)
    tp = len(pred_set & ref_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(ref_set) if ref_set else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def has_hf_weights(hf_dir: Path) -> bool:
    if not hf_dir.exists():
        return False
    # Check for safetensors or pytorch weights
    for fname in os.listdir(hf_dir):
        if fname.startswith("model.safetensors") or fname.startswith("pytorch_model"):
            return True
    return False


def ensure_hf_model_from_actor(actor_dir: Path, target_dir: Path, trust_remote_code: bool) -> Path:
    """
    Ensure we have an HF-loadable model directory for this checkpoint.
    Priority:
      1) If actor/huggingface already has weights, use it
      2) Else, try to merge FSDP shards into `target_dir` using VERL merger
    """
    hf_dir = actor_dir / "huggingface"
    if has_hf_weights(hf_dir):
        return hf_dir

    # Try in-process merger if available
    if FSDPModelMerger is not None and ModelMergerConfig is not None:
        cfg = ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            local_dir=str(actor_dir),
            target_dir=str(target_dir),
            hf_upload_path=None,
            private=False,
            test_hf_dir=None,
            tie_word_embedding=False,
            trust_remote_code=bool(trust_remote_code),
            is_value_model=False,
            use_cpu_initialization=False,
        )
        merger = FSDPModelMerger(cfg)
        merger.merge_and_save()
        return target_dir

    # Fallback: call merger via subprocess to avoid import issues
    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(target_dir),
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Verify weights exist after merge
        if has_hf_weights(target_dir):
            return target_dir
        raise RuntimeError(
            f"Model merger ran but no HF weights found in {target_dir}. stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Failed to merge FSDP checkpoints to HuggingFace format via subprocess.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
        ) from e


@torch.no_grad()
def generate_answers(
    model_dir: Path,
    questions: Iterable[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    trust_remote_code: bool,
) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    outputs: List[str] = []
    do_sample = temperature > 0.0

    for q in questions:
        if not isinstance(q, str):
            q = "" if q is None else str(q)
        inputs = tokenizer(q, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 0.0),
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        # Heuristic: strip the original prompt prefix if present
        if text.startswith(q):
            text = text[len(q) :].strip()
        outputs.append(text)
    return outputs


def evaluate_step(
    step: int,
    checkpoint_root: Path,
    dataset: pd.DataFrame,
    question_col: str,
    answer_col: str,
    answer_json_col: str | None,
    answer_json_key: str | None,
    out_dir: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    trust_remote_code: bool,
    save_jsonl: bool,
) -> dict:
    actor_dir = checkpoint_root / f"global_step_{step}" / "actor"
    merged_dir = out_dir / "merged_models" / f"step_{step}"
    merged_dir.mkdir(parents=True, exist_ok=True)

    model_dir = ensure_hf_model_from_actor(actor_dir, merged_dir, trust_remote_code=trust_remote_code)

    preds = generate_answers(
        model_dir=model_dir,
        questions=dataset[question_col].tolist(),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        trust_remote_code=trust_remote_code,
    )

    # Prepare references (answers)
    if answer_col is not None and answer_col in dataset.columns:
        refs = dataset[answer_col].astype(str).tolist()
    elif answer_json_col is not None and answer_json_key is not None:
        import json as _json

        def _extract_answer(x):
            if isinstance(x, dict):
                return str(x.get(answer_json_key, ""))
            if isinstance(x, str):
                try:
                    obj = _json.loads(x)
                    if isinstance(obj, dict):
                        return str(obj.get(answer_json_key, ""))
                except Exception:
                    return ""
            return ""

        if answer_json_col not in dataset.columns:
            raise ValueError(
                f"answer_json_col '{answer_json_col}' not in dataset. Available: {list(dataset.columns)}"
            )
        refs = dataset[answer_json_col].apply(_extract_answer).tolist()
    else:
        raise ValueError(
            "No valid answers source. Provide --answer_col that exists in dataset, "
            "or both --answer_json_col and --answer_json_key to extract answers."
        )

    rows: List[dict] = []
    f1_list: List[float] = []
    rouge_l_list: List[float] = []
    em_list: List[float] = []
    has_data_source = "data_source" in dataset.columns
    for i, (q, r, p) in enumerate(zip(dataset[question_col].tolist(), refs, preds)):
        f1 = token_f1(p, r)
        rouge = rouge_l_f1(p, r)
        em = float(p.strip() == r.strip())
        f1_list.append(f1)
        rouge_l_list.append(rouge)
        em_list.append(em)
        row = {
            "step": step,
            "sample_id": i,
            "question": q,
            "prediction": p,
            "answer": r,
            "token_f1": f1,
            "rougeL_f1": rouge,
            "exact_match": em,
        }
        if has_data_source:
            try:
                row["data_source"] = dataset["data_source"].iloc[i]
            except Exception:
                pass
        rows.append(row)

    report_df = pd.DataFrame(rows)
    report_path = out_dir / f"report_step_{step}.parquet"
    report_df.to_parquet(report_path, index=False)

    # Optional JSONL exports
    if save_jsonl:
        import json as _json
        step_jsonl = out_dir / f"predictions_step_{step}.jsonl"
        with open(step_jsonl, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(_json.dumps(row, ensure_ascii=False) + "\n")
        # Append to combined jsonl
        combined_jsonl = out_dir / "predictions_all_steps.jsonl"
        with open(combined_jsonl, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(_json.dumps(row, ensure_ascii=False) + "\n")

    agg = {
        "step": step,
        "num_samples": len(report_df),
        "token_f1": float(sum(f1_list) / max(len(f1_list), 1)),
        "rougeL_f1": float(sum(rouge_l_list) / max(len(rouge_l_list), 1)),
        "exact_match": float(sum(em_list) / max(len(em_list), 1)),
        "report_path": str(report_path),
    }
    with open(out_dir / f"metrics_step_{step}.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    return agg


def main():
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root)
    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_root.exists():
        print(f"[ERROR] checkpoint_root not found: {checkpoint_root}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.dataset_path):
        print(f"[ERROR] dataset_path not found: {args.dataset_path}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_parquet(args.dataset_path)
    # Validate question column
    if args.question_col not in df.columns:
        print(f"[ERROR] Column '{args.question_col}' not in dataset. Available: {list(df.columns)}", file=sys.stderr)
        sys.exit(3)

    # Validate answer source: either a direct column or a JSON column+key
    answer_source_ok = False
    if args.answer_col and args.answer_col in df.columns:
        answer_source_ok = True
    elif args.answer_json_col and args.answer_json_key:
        if args.answer_json_col in df.columns:
            answer_source_ok = True
        else:
            print(
                f"[ERROR] answer_json_col '{args.answer_json_col}' not in dataset. Available: {list(df.columns)}",
                file=sys.stderr,
            )
            sys.exit(3)
    if not answer_source_ok:
        print(
            "[ERROR] No valid answer source. Provide --answer_col that exists in dataset, "
            "or both --answer_json_col and --answer_json_key.",
            file=sys.stderr,
        )
        sys.exit(3)
    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)

    summary_rows: List[dict] = []
    # If saving combined JSONL, clear file first (fresh run folder already timestamped, but double safety)
    if args.save_jsonl:
        combined_jsonl = out_dir / "predictions_all_steps.jsonl"
        if combined_jsonl.exists():
            combined_jsonl.unlink()

    for step in steps:
        print(f"==> Evaluating step {step}")
        agg = evaluate_step(
            step=step,
            checkpoint_root=checkpoint_root,
            dataset=df,
            question_col=args.question_col,
            answer_col=args.answer_col,
            answer_json_col=args.answer_json_col,
            answer_json_key=args.answer_json_key,
            out_dir=out_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            trust_remote_code=args.trust_remote_code,
            save_jsonl=args.save_jsonl,
        )
        print(
            f"[step {step}] EM={agg['exact_match']:.4f} | token_F1={agg['token_f1']:.4f} | ROUGE-L_F1={agg['rougeL_f1']:.4f} | n={agg['num_samples']}"
        )
        summary_rows.append(agg)

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    print(f"Done. Summary saved to: {out_dir/'summary.csv'}")


if __name__ == "__main__":
    main()


