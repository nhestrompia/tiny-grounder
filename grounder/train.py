"""Train a grounder.

uv run python -m grounder.train --config configs/m.yaml --pages pages_a pages_b --train train.jsonl \
    --dev dev.jsonl --tokenizer tok.json --out runs/m
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from grounder.data.dataset import AugmentConfig, BucketBatchSampler, GrounderDataset, collate
from grounder.data.pages import load_examples, load_pages_from
from grounder.features import MODEL_INPUTS
from grounder.eval import evaluate, rank_model, select_score
from grounder.model import Grounder, GrounderConfig, listwise_loss, save_checkpoint
from grounder.tokenizer import load_tokenizer


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to zero."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def make_optimizer(model: Grounder, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """Weight decay on matrices only (not biases, norms, NULL vector)."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=lr
    )


def _amp(device: torch.device) -> tuple[torch.dtype | None, bool]:
    """(autocast dtype, needs grad scaler); mixed precision only on CUDA."""
    if device.type != "cuda":
        return None, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    tokenizer = load_tokenizer(args.tokenizer)
    cfg = GrounderConfig.from_yaml(args.config)
    if tokenizer.get_vocab_size() > cfg.vocab:
        raise ValueError(f"tokenizer vocab {tokenizer.get_vocab_size()} exceeds config vocab {cfg.vocab}")
    pages = load_pages_from(args.pages)
    train_ex, dev_ex = load_examples(args.train), load_examples(args.dev)

    limits = {"max_q": cfg.max_q, "max_text": cfg.max_text, "max_ctx": cfg.max_ctx}
    dataset = GrounderDataset(pages, train_ex, tokenizer, AugmentConfig(), args.seed, **limits)
    sampler = BucketBatchSampler(dataset.sizes(), args.batch_size, shuffle=True, seed=args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=args.workers)

    model = Grounder(cfg).to(device)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    total_steps = min(args.epochs * len(sampler), args.max_steps or math.inf)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, args.warmup, int(total_steps))
    )
    amp_dtype, use_scaler = _amp(device)
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    print(f"{model.count_params():,} params on {device}; {len(dataset)} train examples, {int(total_steps)} steps")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_file = open(out / "metrics.jsonl", "w", encoding="utf-8")
    best, step = -1.0, 0
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        model.train()
        started, losses = time.time(), []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(*(batch[k] for k in MODEL_INPUTS))
            loss = listwise_loss(logits, batch["positive"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            step += 1
            if step % args.log_every == 0:
                print(f"step {step} loss {np.mean(losses[-args.log_every:]):.4f}")
            if step >= total_steps:
                break

        dev = evaluate(dev_ex, rank_model(model, tokenizer, pages, dev_ex, args.eval_batch_size))
        score = select_score(dev)
        row = {"epoch": epoch, "step": step, "train_loss": float(np.mean(losses)), "select_top1": score,
               "dev": dev, "seconds": round(time.time() - started, 1)}
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()
        print(f"epoch {epoch} loss {row['train_loss']:.4f} dev top-1 {score:.3f} ({row['seconds']}s)")
        if score > best:
            best = score
            save_checkpoint(model, out / "model.pt", args.tokenizer)
        if step >= total_steps:
            break
    metrics_file.close()
    return out / "model.pt"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--pages", nargs="+", required=True, help="page directories, merged by page_id")
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="auto", help="auto (cuda if available, else cpu), cpu, cuda, mps")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
