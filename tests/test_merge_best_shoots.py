"""Тесты сопоставления разворотов двух съёмок и выбора кадра (merge_best_shoots.py)."""

from pathlib import Path

import numpy as np

from merge_best_shoots import Frame, Shoot, align, choose, output_name


def make_frame(name: str, dom: float | None) -> Frame:
    """Кадр без признаков — для тестов выбора и имён.

    Args:
        name: Имя RAF-файла.
        dom: Балл DOM.

    Returns:
        Frame с пустыми признаками.
    """
    return Frame(Path("/x") / name, np.zeros((0, 2), np.float32), np.zeros((0, 61), np.uint8), dom)


def test_align_with_gaps_on_both_sides():
    """Новая съёмка начинается с двух лишних кадров (обложки), в старой — лишний кадр в середине."""
    # Старая: A B X C D; новая: o1 o2 A B C D
    sim = np.zeros((5, 6))
    for i, j in [(0, 2), (1, 3), (3, 4), (4, 5)]:
        sim[i, j] = 0.4
    # Слабое ложное сходство не должно стать парой
    sim[2, 4] = 0.02
    steps = align(sim, min_sim=0.05)
    assert steps == [(None, 0), (None, 1), (0, 2), (1, 3), (2, None), (3, 4), (4, 5)]


def test_choose_prefers_new_on_tie_and_old_when_clearly_sharper():
    """При почти равных баллах берётся новый, при заметно лучшем старом — старый."""
    old, new = make_frame("DSCF0010.RAF", 3.104), make_frame("DSCF0012.RAF", 3.101)
    assert choose([(Shoot.OLD, old), (Shoot.NEW, new)], tie=0.005)[0] is Shoot.NEW
    old_sharp = make_frame("DSCF0010.RAF", 3.120)
    assert choose([(Shoot.OLD, old_sharp), (Shoot.NEW, new)], tie=0.005)[0] is Shoot.OLD


def test_output_name_keeps_blurry_marks():
    """Пометки .blurry.* старой съёмки сохраняются, постфикс съёмки — через точку."""
    frame = make_frame("DSCF0022.blurry.defocus_medium.RAF", 2.946)
    assert output_name(14, frame, Shoot.OLD) == "014_DSCF0022.blurry.defocus_medium.старый.RAF"


def test_output_name_sub_index_sorts_inside_spread():
    """Кадры одного разворота с подындексами встают по имени между соседними разворотами, старый первым."""
    old = output_name(117, make_frame("DSCF0122.RAF", 2.92), Shoot.OLD, sub=1)
    new = output_name(117, make_frame("DSCF0127.RAF", 3.02), Shoot.NEW, sub=2)
    assert old == "117.1_DSCF0122.старый.RAF"
    assert sorted(["118_DSCF0128.новый.RAF", new, "116_DSCF0126.новый.RAF", old]) == [
        "116_DSCF0126.новый.RAF",
        old,
        new,
        "118_DSCF0128.новый.RAF",
    ]
