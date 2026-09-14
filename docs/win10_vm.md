# Windows-VM с FineReader

Гостевая Windows 10 22H2 в VMware Workstation: `~/vmware/win10/win10.vmx`, NAT vmnet8, гость
`192.168.116.128`, диск 400 GiB (уменьшен с 700 в сентябре 2026; бэкап
`~/vmware/win10.bak.20260909`, 140 GiB, удалить, когда станет не нужен). Внутри — ABBYY
FineReader PDF 16.0.14 (`C:\Program Files\ABBYY FineReader 16`).

## FineReader: только Hot Folder

- **Командная строка не работает**: ключи лицензируются (`FineReader|Feature|CommandLine`).
  `FineReader.exe <файл> /lang Russian /out <pdf> /quit` открывает картинку в PDF Editor и
  висит — процесс приходится убивать. В бинарнике есть только `/noSplash`, `/recognize`, `/untitled`.
- **`HotFolder.exe`** запускается полноценно: задача настраивается в GUI, дальше разбирает
  папку сама. Так распознаются промежуточные PDF пака (одно задание на папку, поэтому PDF
  собираются по паку, а не по годам). Два задания: полные PDF с распрямлением строк и
  `pages_with_pics_only` без него. Опцию «Исправлять ориентацию страницы» не включать —
  валит прогон на боковых иллюстрациях.
- Квоты Hot Folder заданы лицензией, в реестре их нет — смотреть в окне Hot Folder.
  `HotFolderCoreCount` может резать параллелизм сильнее 16 ядер хоста.
- Выгрузка в DOCX («форматированный текст») не сохраняет соответствие страниц: привязывать
  кусок DOCX к скану только через текстовый слой распознанного PDF (редкие токены, окно ±3
  страницы; `research/legacy/table_processing/mining/page_match.py`).

## Доступ с хоста

VMware Tools живы, RDP и SSH в госте закрыты, SMB открыт. Рабочий путь — guest ops:

```bash
vmrun -T ws -gu admin -gp '<пароль>' listDirectoryInGuest  ~/vmware/win10/win10.vmx 'C:\Users\admin'
vmrun -T ws -gu admin -gp '<пароль>' copyFileFromHostToGuest ~/vmware/win10/win10.vmx <src> 'C:\...'
vmrun -T ws -gu admin -gp '<пароль>' runProgramInGuest       ~/vmware/win10/win10.vmx \
    'C:\Windows\System32\cmd.exe' '/c ver > C:\Users\admin\out.txt'
```

- Учётка `admin`; пароль у пользователя, в репо и памяти не хранится.
- Всю команду cmd передавать **одним** аргументом; `/c` отдельным аргументом даёт exit 1.
  Вывод в CP866.
- **Прав администратора guest ops не дают** (отфильтрованный токен): планировщик задач,
  `diskpart`, `chkdsk /f`, `Start-Process -Verb RunAs` — отказ 0x80070005. Всё, что требует
  UAC, класть в гостя `.bat` и просить пользователя запустить «от имени администратора».
- «Быстрая вставка» в консоли: клик по окну переводит его в режим выделения и замораживает
  выполнение до Esc руками.
- Общие папки в гостя: `/mnt/dump3` → `DUMP`, `/mnt/SYSTEM` → `system`. Обмен файлами через
  них обычно проще guest ops.
- Скрипты для гостя — `tools/win_vm/` (PowerShell/bat, вне git).

## Уменьшение vmdk (сделано 2026-09-09, на случай повторения)

`vmware-vdiskmanager` умеет только расширять (`-x`) и уплотнять (`-k`). `sudo` на хосте
требует пароль, значит нет `qemu-nbd`/loop; guest ops без админа — нет `diskpart`. Путь:

1. Раздел вынимается из vmdk оффсетным view:
   `qemu-img convert --image-opts driver=raw,offset=<start*512>,size=<len*512>,file.driver=vmdk,file.file.driver=file,file.file.filename=<vmdk>`
   (два уровня `file.file`, иначе «A block device must be specified for "file"»).
2. `ntfsresize -s <байт>` на вынутом образе, `truncate`.
3. Дескриптор vmdk правится текстом: убрать хвостовые экстенты `RW ... SPARSE "win10-sNNN.vmdk"`
   и их файлы, `ddb.geometry.cylinders = capacity/(255*63)`.
4. Обратная запись — тот же view как `--target-image-opts` с `qemu-img convert -n`; GPT
   (34 сектора в начале, 33 в конце) писать с `-S 0`.
5. После записи vmdk раздувается до полной ёмкости — `vmware-vdiskmanager -k` (401 → 140 GiB).
6. `ntfsresize` помечает том грязным; autochk при загрузке не срабатывает — `chkdsk C: /F`
   через `.bat` от администратора.
