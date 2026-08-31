#!/usr/bin/env bash
#
# Прогон detect_and_crop на партии «1972 готово» (620 JPG в 12 подпапках):
# детекция разворота → поворот → кроп с припусками, апскейл ×2, компенсация
# уровней, сохранение в PNG. Debug-оверлеи (JPEG) — рядом, в отдельной папке.
#
set -euo pipefail

# Скрипт лежит в run_scripts/<подсистема>/, а пути внутри отсчитываются от корня
# репозитория — поднимаемся на два уровня.
cd "$(dirname "$0")/../.."

INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/пак-1 1966-1976 (300 DPI)/1976/09"
OUTPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/нарезка сканов/пак-1 (1966-1976)/cropped/1976/09"
DEBUG_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/нарезка сканов/пак-1 (1966-1976)/debug/1976/09"

echo "detect_and_crop:"
echo "  input  = $INPUT_DIR"
echo "  output = $OUTPUT_DIR"
echo "  debug  = $DEBUG_DIR"

# Ложные «пальцы» на печатном контенте. По умолчанию включён
# --drop-inner-finger-cores: ядра маски пальца, целиком лежащие в глубине кадра
# (напечатанный портрет, принятый детектором за кожу), выбрасываются ДО дилатации —
# иначе она склеивает их с настоящим пальцем в один компонент, тот проходит краевую
# проверку за счёт соседа, и LaMa затирает фотографию (так был испорчен портрет на
# IMG_0011 из 1975/10: --text-protect-mode=copy-back-layout-zones не спас, потому что
# Surya не нашла на этом фото ни одного блока layout и возвращать было нечего).
#
# Чтобы ВЫКЛЮЧИТЬ и вернуть прежнее поведение (краевая проверка по уже раздутой
# маске), добавьте в список аргументов ниже строку:
#     --no-drop-inner-finger-cores \
# Имеет смысл, только если детектор рвёт силуэт настоящего пальца и дальний фрагмент
# (кончик) не достаёт до края кадра — с включённым флагом такой фрагмент отбрасывается
# вместе с ложными. Признак в логе: «ядер не у края убрано=N» на кадрах без портретов.

uv run python -m ocr_utils.scan_cropping \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --debug-dir "$DEBUG_DIR" \
    --recursive \
    --top-margin -120 \
    --bottom-margin -180 \
    --left-margin -230 \
    --right-margin -230 \
    --output-format tiff \
    --force-dpi=300 \
    --compensate-levels \
    --finger-dilate-px=60 \
    --max-asymmetric-dilation-ratio=2 \
    --finger-zone-light-increment=20,20 \
    --extra-erosion-px=110 \
    --protect-text-layout \
    --text-protect-mode=copy-back-layout-zones \
    --layout-pad-px=12,48 \
    --bg-fill-method=nearest \
    --bg-fill-blur-px=16 \
    --log-level=INFO \
    --remove-fingers \
    --crop-mode=pixel-exact \
    --crop-fill-method=replicate \
    --crop-fill-blur-px=32 \
    --crop-fill-fade=0.0