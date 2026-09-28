"""Сверка разностных судей (DeepSeek ``free``, tesseract) с разметкой компонентов глазами и с классическим правилом."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.text_block_specks.component_score import load, verdicts
from research.text_block_specks.judge import Verdict, diff_verdict


def texts(path: Path, field: str) -> dict[str, str]:
    """``id вырезки → текст`` из JSONL судьи (``field`` — ``raw`` у DeepSeek, ``text`` у tesseract)."""
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            out[record["id"]] = record[field]
    return out


def judge_table(components: pd.DataFrame, judged: dict[str, str]) -> pd.Series:
    """Вердикт разностного судьи по каждому компоненту (значения :class:`Verdict`; ``missing`` — нет ответа)."""
    out = []
    for row in components.itertuples():
        a, b = judged.get(f"{row.id}_with"), judged.get(f"{row.id}_erased")
        out.append("missing" if a is None or b is None else diff_verdict(a, b, row.end)[0].value)
    return pd.Series(out, index=components.index)


def main(directory: Path, judge_dir: Path) -> None:
    """Таблицы «судья × метка» и каскад «классика, на спорных — судья»."""
    components = load(directory)
    rule = verdicts(components)
    sources = {"tesseract": (judge_dir / "tesseract.jsonl", "text"), "deepseek": (judge_dir / "deepseek" / "free.jsonl", "raw")}
    for name, (path, field) in sources.items():
        if not path.exists():
            continue
        verdict = judge_table(components, texts(path, field))
        components[name] = verdict
        print(f"\n== {name}: вердикт × метка")
        print(pd.crosstab(verdict, components.label))
    components["rule"] = rule
    print("\n== правило × метка")
    print(pd.crosstab(rule, components.label))
    components.to_csv(judge_dir / "verdicts.csv", index=False)


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]), Path(sys.argv[2]))
