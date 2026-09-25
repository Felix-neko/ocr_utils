$OutputEncoding=[Console]::OutputEncoding=[Text.Encoding]::UTF8
"=== ЗАДАЧИ ОЧИСТКИ ==="
Get-ScheduledTask | Where-Object { $_.TaskName -like '*FineReader*' } | ForEach-Object {
  $t=$_
  "{0}    состояние: {1}    от: {2}" -f $t.TaskName, $t.State, $t.Principal.UserId
  $t.Triggers | ForEach-Object {
    $k = $_.CimClass.CimClassName -replace 'MSFT_Task','' -replace 'Trigger',''
    $rep = if ($_.Repetition.Interval) { " повтор каждые " + $_.Repetition.Interval } else { "" }
    "     триггер: $k$rep" }
  $i = Get-ScheduledTaskInfo -TaskName $t.TaskName
  "     последний запуск: {0}   результат: {1}   следующий: {2}" -f $i.LastRunTime, $i.LastTaskResult, $i.NextRunTime
  ""
}
"=== ПАПКА СЕЙЧАС ==="
$d='C:\Users\admin\AppData\Local\Temp\ABBYY\FineReader\16\PDFTransformer'
if (Test-Path $d) {
  $f=Get-ChildItem $d -File -Recurse -Force
  "файлов {0}, {1:N2} ГБ" -f $f.Count, (($f|Measure-Object Length -Sum).Sum/1GB)
} else { "папки нет (FineReader создаст сам)" }
"процессов ABBYY: " + (Get-Process FineReader,HotFolder,OcrEngine.Background.Host -EA SilentlyContinue|Measure-Object).Count
""
"=== ЛОГ ОЧИСТКИ (20 последних строк) ==="
Get-Content 'C:\Tools\finereader_temp_cleanup.log' -Tail 20
