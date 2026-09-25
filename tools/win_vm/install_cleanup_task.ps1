<#
.SYNOPSIS
    Регистрация задач планировщика для очистки временной папки FineReader.

.DESCRIPTION
    Заводит две задачи, обе от SYSTEM (папка лежит в профиле admin, но SYSTEM имеет к ней
    доступ, и задача отрабатывает даже до входа пользователя):

      1. «При включении» — триггеры At startup И At logon, режим -Full.
         Два триггера, потому что в этой машине включён быстрый запуск (HiberbootEnabled=1):
         после обычного выключения Windows не проходит настоящую загрузку, и один только
         At startup может не сработать. At logon страхует.

      2. «Периодическая» — каждые 15 минут, щадящий режим (без -Full).

    Очистка при ВЫКЛЮЧЕНИИ ставится отдельно и вручную, скриптом это надёжно не делается:
        gpedit.msc → Конфигурация компьютера → Конфигурация Windows → Сценарии
        → Завершение работы → Добавить:
            имя:       powershell.exe
            аргументы: -NoProfile -ExecutionPolicy Bypass -Command "& 'C:\Tools\Clean-FineReaderTemp.ps1' -Full"

.NOTES
    ЗАПУСКАТЬ ОТ АДМИНИСТРАТОРА. Register-ScheduledTask с учётной записью SYSTEM требует
    повышения прав; без него команда отвечает «Отказано в доступе» (HRESULT 0x80070005).
    Запуск через VMware guest operations повышения не даёт — нужен либо админский cmd,
    либо Start-Process -Verb RunAs с подтверждением UAC.

    Скрипт очистки вызывается через -Command, а не через -File. Проверено на этой машине:
    запуск вида «powershell -File Clean-FineReaderTemp.ps1 -DryRun» падает с ошибкой
    разбора аргументов («не удалось преобразовать значение " " в тип System.Int32»),
    а тот же вызов через -Command отрабатывает штатно.

    Файл должен быть сохранён в UTF-8 С BOM (PowerShell 5.1 иначе ломается на русском тексте).
#>

[CmdletBinding()]
param(
    [string]$ScriptPath = 'C:\Tools\Clean-FineReaderTemp.ps1',

    # Период щадящей очистки.
    [int]$IntervalMinutes = 15,

    # Возраст файла для щадящей очистки, минут (по времени создания).
    [int]$MinAgeMinutes = 45
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $ScriptPath)) {
    throw "не найден скрипт очистки: $ScriptPath"
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = ([Security.Principal.WindowsPrincipal]$identity).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    throw ('нужны права администратора: регистрация задачи от SYSTEM без повышения ' +
           'отвечает «Отказано в доступе». Запустите этот скрипт из админского окна ' +
           'или через Start-Process -Verb RunAs.')
}

$startupTask  = 'FineReader temp cleanup (startup)'
$periodicTask = 'FineReader temp cleanup (periodic)'

$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest

# Задачам разрешено стартовать на батарее и не убиваться по таймауту:
# полная очистка десятков гигабайт может идти не одну минуту.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

$argTemplate = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "& ''{0}'' {1}"'


function Register-CleanupTask {
    <#
        Регистрирует задачу и подтверждает результат чтением из планировщика:
        Register-ScheduledTask при отказе в доступе не прерывает выполнение,
        поэтому «зарегистрировано» нужно проверять, а не печатать на веру.
    #>
    param(
        [string]$Name,
        [string]$Arguments,
        [object[]]$Triggers,
        [string]$Description
    )

    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $Arguments

    Unregister-ScheduledTask -TaskName $Name -Confirm:$false -ErrorAction SilentlyContinue

    Register-ScheduledTask -TaskName $Name `
        -Action $action -Trigger $Triggers -Principal $principal -Settings $settings `
        -Description $Description -ErrorAction Stop | Out-Null

    $created = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if (-not $created) {
        throw "задача «$Name» не появилась в планировщике"
    }

    Write-Output "зарегистрирована задача: $Name"
}


Register-CleanupTask -Name $startupTask `
    -Arguments ($argTemplate -f $ScriptPath, '-Full') `
    -Triggers @($(New-ScheduledTaskTrigger -AtStartup), $(New-ScheduledTaskTrigger -AtLogOn)) `
    -Description 'Полная очистка временной папки ABBYY FineReader при включении машины.'

# Повтор навсегда: одиночный триггер с RepetitionInterval.
Register-CleanupTask -Name $periodicTask `
    -Arguments ($argTemplate -f $ScriptPath, "-MinAgeMinutes $MinAgeMinutes") `
    -Triggers @($(New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
                    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes))) `
    -Description 'Щадящая очистка временной папки ABBYY FineReader: незанятые файлы старше порога.'

Write-Output ''
Write-Output "период щадящей очистки: $IntervalMinutes мин, порог возраста: $MinAgeMinutes мин"
Write-Output ''
Write-Output 'ОСТАЛОСЬ ВРУЧНУЮ: сценарий завершения работы в gpedit.msc'
Write-Output '  Конфигурация компьютера -> Конфигурация Windows -> Сценарии -> Завершение работы'
Write-Output '  powershell.exe'
Write-Output ('  ' + ($argTemplate -f $ScriptPath, '-Full'))
