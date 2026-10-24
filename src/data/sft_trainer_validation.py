"""Non-training TRL/Transformers data-path acceptance for pre-encoded SFT rows."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import GPT2Config, GPT2LMHeadModel
from trl import SFTConfig, SFTTrainer

from src.data.sft_collator import SFTDataCollator


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _locked_package_versions(root: Path) -> dict[str, str]:
    wanted = {"transformers", "trl", "datasets", "torch"}
    pinned: dict[str, str] = {}
    for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        name, separator, version = line.partition("==")
        if separator and name.strip().lower() in wanted:
            pinned[name.strip().lower()] = version.strip()
    actual = {name: importlib.metadata.version(name) for name in wanted}
    missing = wanted - set(pinned)
    if missing:
        raise RuntimeError(f"requirements.txt does not pin Trainer data-path dependencies: {sorted(missing)}")
    mismatches = {
        name: {"required": pinned[name], "installed": actual[name]}
        for name in wanted if pinned[name] != actual[name]
    }
    if mismatches:
        raise RuntimeError(f"installed Trainer data-path versions differ from requirements.txt: {mismatches}")
    return actual


def _read_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            result.append(json.loads(line))
            if len(result) == limit:
                break
    if len(result) < limit:
        raise ValueError(f"{path} has fewer than {limit} SFT examples for Trainer data-path validation")
    return result


def _model_state_sha256(model: GPT2LMHeadModel) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate_trainer_data_path_impl(
    dataset_dir: Path,
    *,
    tokenizer: Any,
    pad_token_id: int,
    eos_token_id: int,
    max_sequence_length: int,
    collator: SFTDataCollator | None = None,
    batch_size: int = 2,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Read actual SFT JSONL through pinned SFTTrainer DataLoaders; never train or forward."""
    versions = _locked_package_versions(root)
    train_rows = _read_rows(dataset_dir / "train.jsonl", batch_size)
    dev_rows = _read_rows(dataset_dir / "dev.jsonl", batch_size)
    if collator is None:
        collator = SFTDataCollator(pad_token_id, max_sequence_length=max_sequence_length)

    model_config = GPT2Config(
        vocab_size=max(tokenizer.all_special_ids) + 1,
        n_positions=max_sequence_length,
        n_ctx=max_sequence_length,
        n_embd=8,
        n_layer=1,
        n_head=2,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    # Keep the proof artifact reproducible while exercising real model/Trainer interfaces.
    torch.manual_seed(0)
    model = GPT2LMHeadModel(model_config)
    forward_calls = {"count": 0}

    def count_forward(_module, _inputs):
        forward_calls["count"] += 1

    hook = model.register_forward_pre_hook(count_forward)
    state_before = _model_state_sha256(model)
    with tempfile.TemporaryDirectory(prefix="d043-trainer-contract-") as output_dir:
        args = SFTConfig(
            output_dir=output_dir,
            do_train=False,
            do_eval=False,
            use_cpu=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_checkpointing=False,
            report_to="none",
            save_strategy="no",
            logging_strategy="no",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=None,
            packing=False,
            padding_free=False,
            completion_only_loss=False,
            remove_unused_columns=False,
            train_sampling_strategy="sequential",
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            data_collator=collator,
            train_dataset=Dataset.from_list(train_rows),
            eval_dataset=Dataset.from_list(dev_rows),
            processing_class=tokenizer,
        )
        if not trainer._skip_prepare_dataset:
            raise RuntimeError("SFTTrainer did not honor skip_prepare_dataset=True")
        if trainer.data_collator is not collator:
            raise RuntimeError("SFTTrainer replaced the pre-encoded SFT data collator")
        if trainer.args.max_length is not None or trainer.args.packing or trainer.args.padding_free:
            raise RuntimeError("SFTTrainer data path has truncation, packing, or padding-free enabled")
        if trainer.args.do_train or trainer.args.do_eval:
            raise RuntimeError("Trainer loop flags must stay disabled during D0.4.3")

        train_batch = next(iter(trainer.get_train_dataloader()))
        dev_batch = next(iter(trainer.get_eval_dataloader()))
        expected_train = collator(train_rows)
        expected_dev = collator(dev_rows)
        for split, actual, expected in (
            ("train", train_batch, expected_train),
            ("dev", dev_batch, expected_dev),
        ):
            if set(actual) != {"input_ids", "labels", "attention_mask"}:
                raise RuntimeError(f"Trainer {split} batch exposes unexpected model-forward fields: {sorted(actual)}")
            for name in ("input_ids", "labels", "attention_mask"):
                if not torch.equal(actual[name].cpu(), expected[name].cpu()):
                    raise RuntimeError(f"Trainer {split} DataLoader changed {name} from the custom collator")
            if actual["labels"].shape != actual["input_ids"].shape:
                raise RuntimeError(f"Trainer {split} DataLoader returned mismatched label/input shapes")
        train_columns = list(trainer.train_dataset.column_names)
        dev_columns = list(trainer.eval_dataset.column_names)
        train_batch_lengths = [len(row["input_ids"]) for row in train_rows]
        dev_batch_lengths = [len(row["input_ids"]) for row in dev_rows]
        if max(train_batch_lengths + dev_batch_lengths) > max_sequence_length:
            raise RuntimeError("Trainer DataLoader rows exceed S; truncation is forbidden")
        if forward_calls["count"] != 0:
            raise RuntimeError("Trainer data-path verification unexpectedly called model.forward")
        state_after = _model_state_sha256(model)
        if state_before != state_after:
            raise RuntimeError("the tiny validation model changed during the data-path check")
        hook.remove()

    return {
        "status": "passed",
        "trainer": "trl.SFTTrainer",
        "versions": versions,
        "model": {
            "type": "randomly_initialized_tiny_GPT2_structure_only",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "weights_sha256_before": state_before,
            "weights_sha256_after": state_after,
            "forward_calls": forward_calls["count"],
            "training_loop_called": False,
        },
        "dataset_processing_skipped": trainer._skip_prepare_dataset,
        "max_length": trainer.args.max_length,
        "packing": trainer.args.packing,
        "padding_free": trainer.args.padding_free,
        "remove_unused_columns": trainer.args.remove_unused_columns,
        "train_dataset_columns": train_columns,
        "dev_dataset_columns": dev_columns,
        "batch_size": batch_size,
        "train_batch_lengths": train_batch_lengths,
        "dev_batch_lengths": dev_batch_lengths,
        "train_batch_keys": sorted(train_batch),
        "dev_batch_keys": sorted(dev_batch),
        "train_labels_equal_custom_collator": True,
        "dev_labels_equal_custom_collator": True,
        "attention_mask_equal_custom_collator": True,
        "no_sample_metadata_passed_to_model": not (set(train_batch) | set(dev_batch)) - {
            "input_ids", "labels", "attention_mask"
        },
    }


def validate_trainer_data_path(
    dataset_dir: Path,
    *,
    tokenizer: Any,
    pad_token_id: int,
    eos_token_id: int,
    max_sequence_length: int,
    collator: SFTDataCollator | None = None,
    batch_size: int = 2,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run the real data-path check without perturbing caller RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return _validate_trainer_data_path_impl(
            dataset_dir,
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_sequence_length=max_sequence_length,
            collator=collator,
            batch_size=batch_size,
            root=root,
        )
