# restart.ps1 — Python AI Engine 完全重启脚本
#
# 背景坑（docs/dev-improvement-plan.md 项9）：
#   1. 杀 uvicorn 父进程后，multiprocessing 子进程仍占着 :8000 继续 serve 旧代码
#      （netstat 显示父 PID 但 tasklist 已无进程，排查极耗时）
#   2. uvicorn --reload 对子目录（如 app/skill/skills/）变更检测不可靠
# 所以固定用"完全重启"：按命令行特征杀掉整棵进程树 → 端口兜底强杀 → 重新拉起。
#
# 用法（任选其一）：
#   右键 → 使用 PowerShell 运行
#   powershell -ExecutionPolicy Bypass -File restart.ps1
#   pwsh -File restart.ps1

$ErrorActionPreference = 'Stop'
$Port = 8000
$Root = $PSScriptRoot

Write-Host "=== AI Engine 完全重启 (:$Port) ===" -ForegroundColor Cyan

# ── 1. 按命令行特征杀掉整棵 uvicorn 进程树（父进程 + 子进程命令行都含 app.main）──
$targets = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -eq 'python.exe' -and
    $_.CommandLine -like '*digital-human-studio*' -and
    $_.CommandLine -like '*app.main*'
}
foreach ($t in $targets) {
    Write-Host "  停止进程 PID=$($t.ProcessId)"
    Stop-Process -Id $t.ProcessId -Force -ErrorAction SilentlyContinue
}
if (-not $targets) { Write-Host "  未发现运行中的 AI Engine 进程" }

# ── 2. 兜底：端口仍被占用则按占用 PID 强杀（覆盖命令行不含 app.main 的残留子进程）──
$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($procId in ($listeners | Select-Object -ExpandProperty OwningProcess -Unique)) {
    Write-Host "  端口 $Port 仍被占用，强杀 PID=$procId" -ForegroundColor Yellow
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
}

# ── 3. 等待端口释放（最多 5 秒）──
for ($i = 0; $i -lt 10; $i++) {
    if (-not (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) { break }
    Start-Sleep -Milliseconds 500
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Write-Error "端口 $Port 无法释放，请手动检查（netstat -ano | findstr :$Port）"
    exit 1
}
Write-Host "  端口 $Port 已释放" -ForegroundColor Green

# ── 4. 前台拉起（完全重启，不用 --reload；Ctrl+C 停止）──
Set-Location $Root
$env:PYTHONIOENCODING = 'utf-8'

$uvicorn = Join-Path $Root '.venv\Scripts\uvicorn.exe'
Write-Host "  启动 AI Engine ..." -ForegroundColor Cyan
if (Test-Path $uvicorn) {
    & $uvicorn app.main:app --host 0.0.0.0 --port $Port
} else {
    # 无本地 venv 时回退到 uv
    uv run uvicorn app.main:app --host 0.0.0.0 --port $Port
}
