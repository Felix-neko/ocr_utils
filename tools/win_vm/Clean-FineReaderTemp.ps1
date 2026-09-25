<#
.SYNOPSIS
    Очистка временной папки ABBYY FineReader (PDFTransformer).

.DESCRIPTION
    FineReader не освобождает временные файлы завершённых PDF до конца пакетного задания.
    Замер на живом джобе: папка растёт на ~2,2-2,5 ГБ на каждую обработанную PDF и ничего
    не отдаёт обратно. На пакете в 246 PDF это ~550 ГБ — диск кончается раньше, чем джоб.

    Два режима.

    -Full  — снести содержимое подчистую. Безопасно ТОЛЬКО когда FineReader не запущен:
             при выключенном FineReader занятых файлов в папке 0 из 466, всё это сироты.
             Если процессы ABBYY живы, режим ничего не трогает и выходит.

    без -Full (по умолчанию) — щадящий режим, рассчитанный на работу ВО ВРЕМЯ джоба.
             Файл удаляется, только если выполнены все три условия:
               * файл не занят другим процессом;
               * он старше -MinAgeMinutes ПО ВРЕМЕНИ СОЗДАНИЯ;
               * он не служебный (см. $SERVICE_FILES).

.NOTES
    Возраст считается по CreationTime, и это принципиально. LastWriteTime здесь врёт:
    FineReader копирует исходную PDF вместе с её меткой времени, поэтому у свежесозданного
    временного файла время изменения может показывать вчерашнюю ночь. Политика «удалять всё
    старше N минут по времени изменения» снесла бы активные файлы в первую же минуту.

    Одной проверки возраста тоже мало: файлы уже завершённых PDF FineReader держит открытыми
    ещё десятки минут. Поэтому занятость проверяется явно — попыткой открыть файл монопольно
    на чтение. Содержимое при этом не меняется.

    Запускать через -Command, а не через -File: на этой машине вызов вида
    «powershell -File Clean-FineReaderTemp.ps1 -DryRun» падает с ошибкой разбора аргументов,
    а через -Command отрабатывает штатно.

    Файл должен быть сохранён в UTF-8 С BOM: PowerShell 5.1 иначе читает русский текст
    как ANSI и падает на разборе.
#>

[CmdletBinding()]
param(
    # Полная очистка. Только при неработающем FineReader.
    [switch]$Full,

    # Минимальный возраст файла (по времени СОЗДАНИЯ) для щадящего режима.
    # Одна PDF обрабатывается 8-15 минут, 45 минут — запас втрое.
    [int]$MinAgeMinutes = 45,

    # Ничего не удалять, только показать и записать в лог, что было бы удалено.
    [switch]$DryRun,

    [string]$Path = 'C:\Users\admin\AppData\Local\Temp\ABBYY\FineReader\16\PDFTransformer',

    [string]$LogPath = 'C:\Tools\finereader_temp_cleanup.log'
)

$ErrorActionPreference = 'SilentlyContinue'

# Служебные файлы текущего документа. Основная их масса — .frdat в подпапке PDFTCM*
# (замер: 1026 файлов на 864 МБ для одного выпуска), плюс мелочь рядом. Живут ровно
# столько, сколько обрабатывается документ, и уходят вместе с его подпапкой, так что
# это рабочий объём, а не утечка. По возрасту щадящий режим их и так не тронет — маска
# нужна на случай особенно долгого документа, чтобы не сломать ему состояние ради
# нескольких сотен мегабайт.
$SERVICE_FILES = @('*.frdat', 'documentSynthesisData.*', 'delStyles.dat', 'docGDS.dat')

# Процессы, чьё присутствие означает «FineReader работает».
$ABBYY_PROCESSES = @('FineReader', 'HotFolder', 'OcrEngine.Background.Host')


function Write-Log {
    param([string]$Message)

    $line = '{0:yyyy-MM-dd HH:mm:ss}  {1}' -f (Get-Date), $Message
    $dir = Split-Path $LogPath -Parent
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    Add-Content -Path $LogPath -Value $line -Encoding UTF8
    Write-Output $line
}


function Test-FileLocked {
    <#
        Занят ли файл другим процессом. Открываем монопольно на чтение и сразу закрываем —
        содержимое файла не меняется. Успех означает, что файл никем не удерживается.
    #>
    param([string]$FilePath)

    try {
        $stream = [System.IO.File]::Open($FilePath, 'Open', 'Read', 'None')
        $stream.Close()
        $stream.Dispose()
        return $false
    } catch {
        return $true
    }
}


function Test-ServiceFile {
    param([string]$Name)

    foreach ($pattern in $SERVICE_FILES) {
        if ($Name -like $pattern) { return $true }
    }
    return $false
}


# --- основной ход ---

if (-not (Test-Path $Path)) {
    Write-Log "папка не найдена: $Path — очищать нечего"
    exit 0
}

$running = @(Get-Process -Name $ABBYY_PROCESSES -ErrorAction SilentlyContinue).Count
$mode = if ($Full) { 'ПОЛНАЯ' } else { "щадящая (старше $MinAgeMinutes мин)" }
$dry = if ($DryRun) { ' [ХОЛОСТОЙ ХОД]' } else { '' }

Write-Log "старт: режим $mode$dry, процессов ABBYY: $running"

if ($Full -and $running -gt 0) {
    Write-Log "ОТМЕНА: запрошена полная очистка, но FineReader работает ($running процессов). Ничего не тронуто."
    exit 0
}

$deleted = 0
$freedBytes = 0L
$skippedLocked = 0
$skippedYoung = 0
$skippedService = 0
$now = Get-Date

foreach ($file in Get-ChildItem -Path $Path -File -Recurse -Force) {

    if (-not $Full) {
        if (Test-ServiceFile $file.Name) { $skippedService++; continue }

        if (($now - $file.CreationTime).TotalMinutes -lt $MinAgeMinutes) { $skippedYoung++; continue }

        if (Test-FileLocked $file.FullName) { $skippedLocked++; continue }
    }

    $size = $file.Length

    if ($DryRun) {
        $deleted++
        $freedBytes += $size
        continue
    }

    Remove-Item -LiteralPath $file.FullName -Force -ErrorAction SilentlyContinue

    if (Test-Path -LiteralPath $file.FullName) {
        # Не удалился — значит всё-таки занят. Это второй предохранитель поверх Test-FileLocked.
        $skippedLocked++
    } else {
        $deleted++
        $freedBytes += $size
    }
}

# Пустые подпапки (FineExec, PDFTCM*) убираем только при полной очистке:
# во время работы они ещё нужны FineReader.
if ($Full -and -not $DryRun) {
    Get-ChildItem -Path $Path -Directory -Force | ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
}

$freedGb = [math]::Round($freedBytes / 1GB, 2)
$verb = if ($DryRun) { 'удалилось бы' } else { 'удалено' }

Write-Log ("итог: $verb файлов $deleted, освобождено $freedGb ГБ; " +
           "пропущено занятых $skippedLocked, свежих $skippedYoung, служебных $skippedService")
