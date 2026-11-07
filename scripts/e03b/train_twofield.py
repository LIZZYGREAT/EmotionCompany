"""E03-B-P2 two-field QLoRA training (independent entry, no D0.3 coupling).

Reads the pre-encoded jsonl (input_ids/labels) produced for this experiment,
trains with 4-bit NF4 + LoRA, and saves checkpoints at fixed step intervals.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

PAD_ID = 151643
EOS_ID = 151645


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.rows = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                self.rows.append(json.loads(line))
        self.rows.sort(key=lambda r: len(r["input_ids"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
        }


class CompletionTrainer(Trainer):
    """只在 completion 段算 loss，logits 只取末尾 k 个位置，省显存。"""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        completion_length = int(inputs.pop("completion_length")[0])
        k = min(completion_length + 1, labels.size(1))
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            logits_to_keep=k,
        )
        logits = outputs.logits
        target = labels[:, -k + 1:]
        mask = target.ne(-100)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=-100,
        )
        if return_outputs:
            return loss, outputs
        return loss


class PadCollator:
    def __init__(self, pad_id: int = PAD_ID):
        self.pad_id = pad_id
        self.max_len = 0

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        self.max_len = max(self.max_len, max_len)
        input_ids, labels, attention = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            input_ids.append(torch.cat([f["input_ids"], torch.full((pad,), self.pad_id, dtype=torch.long)]))
            labels.append(torch.cat([f["labels"], torch.full((pad,), -100, dtype=torch.long)]))
            attention.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = PAD_ID

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    dataset = EncodedDataset(args.data)
    print("rows", len(dataset))
    first = dataset[0]
    prompt_len = int((first["labels"] == -100).sum())
    print("first_prompt_length", prompt_len)
    print("first_label_head", first["labels"][:5].tolist())
    print("first_tail", first["input_ids"][-3:].tolist())
    assert first["labels"][:prompt_len].eq(-100).all()
    assert EOS_ID in first["input_ids"].tolist()

    training = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=args.max_steps,
        learning_rate=1e-4,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        bf16=True,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=10,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=dataset,
        data_collator=PadCollator(),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    if not args.dry_run:
        trainer.save_model(os.path.join(args.out, "adapter"))
    print("done", args.max_steps)
    print("max_batch_tokens", trainer.data_collator.max_len)
    print("peak_alloc_gb", round(torch.cuda.max_memory_allocated() / 1e9, 3))
    print("peak_reserved_gb", round(torch.cuda.max_memory_reserved() / 1e9, 3))


if __name__ == "__main__":
    main()
