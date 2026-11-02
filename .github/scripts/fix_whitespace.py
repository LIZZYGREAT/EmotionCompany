#!/usr/bin/env python3
"""Safely normalize trailing whitespace in changed Markdown files."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_SUFFIXES = {".md", ".markdown"}
PROTECTED_PREFIXES = (
    ".git",
    "data/train",
    "data/val",
    "data/processed",
    "docs/private",
    "outputs",
    "models",
    "checkpoints",
    "runs",
    "participant/model",
)
PROTECTED_FILES = {
    "configs/prompts/p0.txt",
    "data/train/train_public.jsonl",
    "data/train/train_public.schema.json",
    "data/val/val_public.jsonl",
    "data/val/val_public.schema.json",
}
DIFF_FILTER = "ACMRTUXB"


class WhitespaceError(RuntimeError):
    """An error that requires the user to resolve an ambiguity."""


class SkippedPath(RuntimeError):
    """A path that is intentionally excluded from automatic processing."""


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise WhitespaceError(detail or f"git {' '.join(args)} failed")
    return result


def git_paths(*args: str) -> set[Path]:
    output = git(*args).stdout
    return {Path(os.fsdecode(value)) for value in output.split(b"\0") if value}


def changed_paths() -> set[Path]:
    return (
        git_paths("diff", "--name-only", "-z", f"--diff-filter={DIFF_FILTER}")
        | git_paths("diff", "--cached", "--name-only", "-z", f"--diff-filter={DIFF_FILTER}")
        | git_paths("ls-files", "--others", "--exclude-standard", "-z")
    )


def staged_paths() -> set[Path]:
    return git_paths("diff", "--cached", "--name-only", "-z", f"--diff-filter={DIFF_FILTER}")


def resolve_repo_path(raw_path: Path) -> tuple[Path, Path]:
    root = ROOT.resolve()
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise WhitespaceError(f"拒绝处理仓库外路径：{raw_path}") from exc

    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise SkippedPath(relative.as_posix())

    resolved = lexical.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise WhitespaceError(f"拒绝处理指向仓库外的路径：{relative.as_posix()}") from exc
    return lexical, relative


def is_protected(relative: Path) -> bool:
    name = relative.as_posix()
    if name in PROTECTED_FILES:
        return True
    return any(
        name == prefix or name.startswith(prefix + "/")
        for prefix in PROTECTED_PREFIXES
    )


def normalize(raw: bytes, relative: Path) -> bytes:
    bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
    payload = raw[len(bom):]
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WhitespaceError(f"{relative.as_posix()}：不是 UTF-8，未修改") from exc
    if "\x00" in text:
        raise WhitespaceError(f"{relative.as_posix()}：疑似二进制文件，未修改")

    crlf_count = payload.count(b"\r\n")
    without_crlf = payload.replace(b"\r\n", b"")
    if b"\r" in without_crlf:
        raise WhitespaceError(f"{relative.as_posix()}：含不支持的 CR 换行，未修改")
    bare_lf_count = payload.count(b"\n") - crlf_count
    if crlf_count and bare_lf_count:
        raise WhitespaceError(f"{relative.as_posix()}：混合 LF/CRLF，未修改")

    newline = "\r\n" if crlf_count else "\n"
    lines = text.split(newline)
    cleaned: list[str] = []
    for number, line in enumerate(lines, start=1):
        content = line.rstrip(" \t")
        trailing = line[len(content):]
        if content and trailing.count(" ") >= 2:
            raise WhitespaceError(
                f"{relative.as_posix()}:{number}：检测到 Markdown 双空格硬换行；"
                "请先显式改写（例如 <br>），脚本未修改该文件"
            )
        if content.endswith("\\") and trailing:
            raise WhitespaceError(
                f"{relative.as_posix()}:{number}：去掉行尾空白会改变反斜杠换行语义，未修改"
            )
        cleaned.append(content)

    while cleaned and not cleaned[-1]:
        cleaned.pop()
    normalized = newline.join(cleaned) + newline
    return bom + normalized.encode("utf-8")


def has_unstaged_difference(relative: Path) -> bool:
    result = git("diff", "--quiet", "--", relative.as_posix(), check=False)
    if result.returncode == 1:
        return True
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise WhitespaceError(detail or f"无法检查暂存文件：{relative.as_posix()}")
    return False


def select_paths(args: argparse.Namespace) -> set[Path]:
    if args.mode == "changed":
        return changed_paths()
    if args.mode == "staged":
        return staged_paths()
    return {Path(args.file)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--changed", action="store_const", const="changed", dest="mode")
    modes.add_argument("--staged", action="store_const", const="staged", dest="mode")
    modes.add_argument("--file", metavar="PATH", help="只处理仓库内的指定 Markdown 文件")
    args = parser.parse_args()
    if args.mode is None:
        args.mode = "file"

    try:
        raw_paths = select_paths(args)
        prepared: list[tuple[Path, Path, bytes]] = []
        for raw_path in sorted(raw_paths, key=lambda item: item.as_posix()):
            try:
                path, relative = resolve_repo_path(raw_path)
            except SkippedPath as exc:
                print(f"跳过符号链接：{exc}")
                continue
            if is_protected(relative):
                print(f"跳过受保护路径：{relative.as_posix()}")
                continue
            if path.suffix.lower() not in MARKDOWN_SUFFIXES:
                print(f"跳过非 Markdown 文件：{relative.as_posix()}")
                continue
            if not path.exists():
                print(f"跳过已删除文件：{relative.as_posix()}")
                continue
            if path.is_symlink():
                raise WhitespaceError(f"拒绝处理符号链接：{relative.as_posix()}")
            if not path.is_file():
                print(f"跳过非普通文件：{relative.as_posix()}")
                continue
            if args.mode == "staged" and has_unstaged_difference(relative):
                raise WhitespaceError(
                    f"{relative.as_posix()} 同时有暂存与未暂存改动；为保护部分暂存内容，"
                    "Hook 未修改或暂存任何文件。请先整理该文件后重试"
                )
            original = path.read_bytes()
            fixed = normalize(original, relative)
            if fixed != original:
                prepared.append((path, relative, fixed))

        # Validate every selected file before writing any, so a refusal is all-or-nothing.
        for path, _relative, fixed in prepared:
            path.write_bytes(fixed)

        if args.mode == "staged":
            for _path, relative, _fixed in prepared:
                git("add", "--", relative.as_posix())

        for _path, relative, _fixed in prepared:
            print(f"已修复：{relative.as_posix()}")
        if not prepared:
            print("没有需要修复的 Markdown 文件。")
        return 0
    except (OSError, WhitespaceError) as exc:
        print(f"whitespace 修复已停止：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
