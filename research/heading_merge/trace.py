"""Трассировка сращиваний: какой шаг детектора склеил стык — смыкание RLSA, сцепка кусков, сборка или слияние рядов."""

from __future__ import annotations

from dataclasses import dataclass, field

from ocr_utils.page_layout.text_blocks import blocks, regroup, segment, zones

# Допуск совпадения края куска (оси) с краем просвета стыка, пиксели рабочей копии.
EDGE_TOL = 4.0


@dataclass
class TraceLog:
    """Журнал событий одного разбора полосы (заполняется подменами :class:`TracePatch`)."""

    blobs: list[tuple[str, object]] = field(default_factory=list)  # (масштаб, stats сгустков RLSA)
    links: list[dict] = field(default_factory=list)  # принятые сцепки кусков
    rows: list[object] = field(default_factory=list)  # ряды ``rows_of``
    merges: list[tuple[str, list, object]] = field(default_factory=list)  # (функция, входные ряды, выходной)


class TracePatch:
    """Обёртки шагов детектора, пишущие журнал; ставятся ПОВЕРХ текущих функций (в том числе подмен стенда).

    Attributes:
        log: Журнал текущего разбора; перед каждой полосой — :meth:`reset`.
    """

    def __init__(self) -> None:
        self.log = TraceLog()
        self._saved: list[tuple[object, str, object]] = []
        self._smeared = segment._smeared
        self._verdict = zones._verdict_of
        self._rows_of = blocks.rows_of
        self._overlapping = regroup.merge_overlapping_pieces
        self._same_line = regroup.merge_same_line

    def reset(self) -> None:
        """Очистить журнал перед новой полосой."""
        self.log = TraceLog()

    def smeared(self, mask, scale, barriers=None, split_rows=segment.SPLIT_ROWS_DEFAULT):
        """Смыкание RLSA: сгустки сохраняются с именем масштаба."""
        count, labels, stats = self._smeared(mask, scale, barriers, split_rows)
        self.log.blobs.append((scale.name, stats.copy()))
        return count, labels, stats

    def verdict(self, left, right, scale, *args, **kwargs):
        """Вердикт сцепки: принятые пары сохраняются с краями и кеглем концов."""
        result = self._verdict(left, right, scale, *args, **kwargs)
        if result is zones.LinkVerdict.ACCEPTED:
            ends = (zones.end_kegl(left, at_start=False), zones.end_kegl(right, at_start=True))
            self.log.links.append(
                {"scale": scale.name, "lx1": left.x1, "rx0": right.x0, "y": (left.cy + right.cy) / 2.0,
                 "lxh": left.x_h, "rxh": right.x_h, "ends": ends}
            )  # fmt: skip
        return result

    def rows_of(self, axes, ink, span, dpi, *args, **kwargs):
        """Ряды колонки сохраняются целиком."""
        rows = self._rows_of(axes, ink, span, dpi, *args, **kwargs)
        self.log.rows.extend(rows)
        return rows

    def _merged(self, name: str, rows: list, out: list) -> None:
        """Записать, какие входные ряды оказались в одном выходном."""
        for row in out:
            parts = [r for r in rows if set(map(id, r.axes)) <= set(map(id, row.axes))]
            if len(parts) > 1:
                self.log.merges.append((name, parts, row))

    def overlapping(self, rows, *args, **kwargs):
        """``regroup.merge_overlapping_pieces`` с записью слияний."""
        out = self._overlapping(rows, *args, **kwargs)
        self._merged("merge_overlapping_pieces", rows, out)
        return out

    def same_line(self, rows, *args, **kwargs):
        """``regroup.merge_same_line`` с записью слияний."""
        out = self._same_line(rows, *args, **kwargs)
        self._merged("merge_same_line", rows, out)
        return out

    def _set(self, module, name: str, value) -> None:
        """Подменить ``module.name`` и запомнить исходное."""
        self._saved.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    def __enter__(self) -> "TracePatch":
        self._set(segment, "_smeared", self.smeared)
        self._set(zones, "_verdict_of", self.verdict)
        self._set(blocks, "rows_of", self.rows_of)
        self._set(regroup, "merge_overlapping_pieces", self.overlapping)
        self._set(regroup, "merge_same_line", self.same_line)
        return self

    def __exit__(self, *exc) -> None:
        for module, name, value in reversed(self._saved):
            setattr(module, name, value)
        self._saved.clear()


def culprits(log: TraceLog, joint: dict, barriers=None) -> list[str]:
    """Шаги, склеившие стык, по журналу разбора полосы.

    Args:
        log: Журнал :class:`TracePatch`.
        joint: Стык (``measure.candidate_joints``): ``x_gap0``, ``x_gap1``, ``y``, ``h_left``, ``h_right``.
        barriers: Линейки-барьеры полосы (``barriers.BarrierLines``) или ``None``: отмечается, проходит ли
            линейка через просвет или рядом с ним (на полкегля внутрь сторон).

    Returns:
        Подписи шагов; пусто — трассировка стык не опознала.
    """
    x0, x1, y = float(joint["x_gap0"]), float(joint["x_gap1"]), float(joint["y"])
    h = max(float(joint["h_left"]), float(joint["h_right"]))
    found = []
    for name, stats in log.blobs:
        # Сгусток, накрывший просвет целиком на высоте стыка.
        left, top, width, height = stats[1:, 0], stats[1:, 1], stats[1:, 2], stats[1:, 3]
        hit = (left <= x0 + 1) & (left + width >= x1 - 1) & (top <= y + h / 2) & (top + height >= y - h / 2)
        if hit.any():
            found.append(f"RLSA[{name}]")
    for link in log.links:
        if abs(link["lx1"] - x0) <= EDGE_TOL and abs(link["rx0"] - x1) <= EDGE_TOL and abs(link["y"] - y) < 1.5 * h:
            ends = tuple(None if e is None else round(e, 1) for e in link["ends"])
            found.append(f"сцепка[{link['scale']}] xh {link['lxh']:.0f}/{link['rxh']:.0f} концы {ends}")
    for row in log.rows:
        spans = sorted((a.x0, a.x1) for a in row.axes)
        if len(spans) < 2 or abs(row.y - y) >= h:
            continue
        reach = spans[0][1]
        for a0, a1 in spans[1:]:
            # Край оси — по её точкам, а стык — по глифам: допуск в полкегля.
            if abs(reach - x0) <= EDGE_TOL + 0.5 * h and abs(a0 - x1) <= EDGE_TOL + 0.5 * h:
                found.append(f"rows_of glyph_h {row.glyph_h:.0f}")
            reach = max(reach, a1)
    for name, parts, row in log.merges:
        if abs(row.y - y) > h:
            continue
        if any(abs(p.x1 - x0) <= EDGE_TOL + 0.5 * h for p in parts) and any(
            abs(p.x0 - x1) <= EDGE_TOL + 0.5 * h for p in parts
        ):
            found.append(f"{name} glyph_h {'/'.join(f'{p.glyph_h:.0f}' for p in parts)}")
    if barriers is not None and not barriers.empty:
        if barriers.crosses((x0 - 0.5 * h, y), (x1 + 0.5 * h, y)):
            found.append("линейка у просвета")
    return found


__all__ = ["TraceLog", "TracePatch", "culprits"]
