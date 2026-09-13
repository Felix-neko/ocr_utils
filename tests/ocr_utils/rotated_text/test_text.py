"""Склейка строк, фильтр текста и фильтр второго мнения."""

from ocr_utils.rotated_text.tables.ocr import CellText, join_lines, strip_junk
from ocr_utils.rotated_text.tables.orientation import looks_like_text
from ocr_utils.rotated_text.tables.second_opinion import accept, disagreement, foreign_script


def test_join_dehyphenates_before_lowercase():
    assert join_lines(["матери-", "ально-техни-", "ческих"]) == "материально-технических"


def test_join_keeps_hyphen_before_capital_and_joins_without_space():
    assert join_lines(["Северо-", "Западный"]) == "Северо-Западный"


def test_join_repeated_dash_stays_single():
    assert join_lines(["1966—1970 гг.", "в % к 1961—", "—1965 гг."]) == "1966—1970 гг. в % к 1961—1965 гг."


def test_join_dash_at_line_end_glues_without_space():
    assert join_lines(["1966—", "1970 гг."]) == "1966—1970 гг."


def test_join_drops_junk_at_edges_and_empty_lines():
    assert join_lines(["| расчетная", "", "лесосека —"]) == "расчетная лесосека"
    assert strip_junk("— в 13 раз |") == "в 13 раз"


def test_looks_like_text():
    assert looks_like_text("расчетная лесосека")
    assert looks_like_text("1790")
    assert looks_like_text("0")
    assert not looks_like_text("")
    assert not looks_like_text("|| — ..")
    assert not looks_like_text("‚=, ^^: 5")


def test_replaceable_short_and_noisy():
    from ocr_utils.rotated_text.tables.pipeline import CellRecord, replaceable

    def record(text, rotate=90):
        return CellRecord(0, 0, 1, 1, False, (0, 0, 1, 1), None, rotate_cw=rotate, text=text)

    assert replaceable(record("расчетная лесосека"), 0.7, 3)[0]
    assert replaceable(record("0"), 0.95, 3)[0]  # ноль на складе — тоже содержимое
    assert replaceable(record("6П13С"), 0.93, 3)[0]
    assert not replaceable(record("6113С"), 0.6, 3)[0]
    assert not replaceable(record("я у у Я 1 т ко Е у я У я В — ы"), 0.53, 3)[0]  # букв много, слова нет
    assert not replaceable(record("ТОТ 08$", 180), 0.75, 3)[0]  # ниже CONFIDENCE_180
    assert replaceable(record("Итого", 180), 0.96, 3)[0]
    assert not replaceable(record("потрё НОСТЬ Г.)"), 0.515, 3)[0]  # ниже CONFIDENCE_REPLACE
    assert not replaceable(record("-Я 3 У м 2 Я я у сво"), 0.66, 3)[0]  # россыпь
    assert replaceable(record("в 13 раз"), 0.9, 3)[0]


def test_foreign_script_and_disagreement():
    assert foreign_script("그는 выделено")
    assert not foreign_script("выделено, 1966 г., ГОСТ-12")
    assert disagreement("расчетная", "расчётная") == 0.0
    assert disagreement("abc", "abd") > 0


def test_accept_second_opinion():
    ours = CellText(lines=["расчетная лесосека"], confidence=0.6)
    assert accept(ours, CellText(lines=["расчётная лесосека"], confidence=0.9))[0]
    assert not accept(ours, CellText(lines=["그는 경기가 расчетная"], confidence=0.9))[0]
    assert not accept(ours, CellText(lines=["совсем другой текст"], confidence=0.9))[0]
    assert not accept(ours, CellText(lines=["расчетная лесосека"], confidence=0.3))[0]  # surya не уверена
    assert not accept(ours, CellText(lines=["расчётная лесосека"], confidence=0.9), ours_reliable=True)[0]
    # tesseract почти ничего не прочёл — согласия не требуется, решают письменность и длина.
    weak = CellText(lines=["рсч"], confidence=0.2)
    assert accept(weak, CellText(lines=["расчет"], confidence=0.8))[0]
    assert not accept(weak, CellText(lines=[""], confidence=0.8))[0]
    doubtful = CellText(lines=["потрё НОСТЬ Г.)"], confidence=0.515)
    assert accept(doubtful, CellText(lines=["Пятидневная потребность III"], confidence=0.72))[0]
    # tesseract уверен, но его чтение само отвергнуто как мусор — согласия не требуется.
    junk = CellText(lines=["6710 [2"], confidence=0.6)
    assert accept(junk, CellText(lines=["с7по 12"], confidence=0.95), ours_plausible=False)[0]
    assert not accept(junk, CellText(lines=["с7по 12"], confidence=0.95), ours_plausible=True)[0]
    # Галлюцинации: зацикленный текст, серия знаков, длинная латиница.
    looped = CellText(lines=["and the second second second second second second second"], confidence=0.99)
    assert not accept(junk, looped, ours_plausible=False)[0]
    assert not accept(junk, CellText(lines=["5 420" + "—" * 40], confidence=0.95), ours_plausible=False)[0]
    latin = CellText(lines=["ale de la company de la company de la company"], confidence=0.96)
    assert not accept(junk, latin, ours_plausible=False)[0]
    long_ok = CellText(
        lines=["Для неслеживающих материалов, требующих защиты от атмосферных осадков и хранения под навесом"],
        confidence=0.99,
    )
    assert accept(junk, long_ok, ours_plausible=False)[0]


def test_cyrillic_homoglyphs():
    from ocr_utils.rotated_text.tables.second_opinion import cyrillic

    assert cyrillic("c<i>13 no 18</i>") == "с13 по 18"
    assert cyrillic("NN n.n.") == "NN n.n."  # N без двойника — ячейка остаётся как есть
    assert cyrillic("Пятидневная потребность III") == "Пятидневная потребность III"
    assert cyrillic("HΠ3 № 20 i=n") == "HΠ3 № 20 i=n"  # буква без двойника — латиница настоящая
    assert cyrillic("a_i, C_kj") == "a_i, C_kj"
