<#
.SYNOPSIS
    多批次"训练 N 局 → 跑一次测试集"的循环实验脚本。

.DESCRIPTION
    每一批做三件事：
      1. eluose.exe --train -i 训练集      训练 $EpisodesPerBatch 局
                                           （局数用编译参数 -DTRAIN_EPISODES 覆盖，不改源码）；
      2. eluose.exe -i 测试集              用贪心策略把测试集跑一遍，记录得分；
      3. 把这一批产生的训练追踪 CSV 另存为 trace_batchNNN.csv，供出图使用。
    合计重复 $BatchCount 批。模型沿用已有 q_weights.dat（脚本会先把它复制到工作
    目录），每一批训练完都会覆盖它，所以后一批是在前一批的基础上继续训练。

    epsilon 退火：整体从 $EpsilonStart 线性降到 $EpsilonEnd，跨所有批次平滑下降。
    每一批都按自己那一段的起点/终点重新编译（-DTRAIN_EPSILON_START / _END），
    批内按局号线性插值；上一批的终点就是下一批的起点，所以整条曲线首尾相接，
    不会每批重新从高处探索一遍。

    训练追踪文件由 eluose.c 配置区里的 TRAINING_TRACE_PATH 决定（默认
    "training_trace.csv"，空串表示不追踪）。脚本按"这一批新出现的 CSV"来认领它，
    因此改不改文件名都能工作；若配置区关掉了追踪，脚本会给出警告、只记测试得分。

    产物都在工作目录（默认 .\runs）里：
      eluose.exe            按本批参数编译出来的可执行文件
      q_weights.dat         持续更新的模型
      trace_batchNNN.csv    每批的逐动作训练追踪
      model_batchNNN.dat    该批测试时用的模型参数快照（135 行，可直接改名为 q_weights.dat 复盘）
      test_scores.csv       每批的测试得分

.EXAMPLE
    .\run_experiments.ps1

    # 训练完出图：
    python .\analyze_training.py .\runs\trace_batch*.csv `
        --test-log .\runs\test_scores.csv --output .\runs\training_analysis.png
#>

$ErrorActionPreference = "Stop"

# ===================== 参数区：按需修改 =====================

# 训练集、测试集（相对脚本所在目录，也可写绝对路径）
$TrainSet = ".\1.txt"
$TestSet  = ".\2.txt"

# 每一批训练多少局
$EpisodesPerBatch = 200

# 训练 → 测试 一共重复多少批
$BatchCount = 20

# epsilon 退火：整体从 $EpsilonStart 平滑降到 $EpsilonEnd，跨 $BatchCount 批分摊
$EpsilonStart = 0.35
$EpsilonEnd   = 0.05

# 工作目录：exe、模型、CSV 都放这里
$WorkDir = ".\runs"

# ============================================================

$repo = $PSScriptRoot
$gcc = "gcc"

function Get-FullPath([string]$path, [string]$what) {
    $full = if ([System.IO.Path]::IsPathRooted($path)) { $path } else { Join-Path $repo $path }
    if (-not (Test-Path -LiteralPath $full)) { throw "找不到$what：$full" }
    return (Resolve-Path -LiteralPath $full).Path
}

# 调用外部程序并捕获 stdout；stderr 照常直接打印，不因编译器警告中断脚本
function Invoke-Captured([string]$exe, [string[]]$arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $exe @arguments | Out-String
        return [pscustomobject]@{ Output = $output; ExitCode = $LASTEXITCODE }
    } finally {
        $ErrorActionPreference = $previous
    }
}

# 列出工作目录里的 CSV 文件名（用于识别这一批新生成的追踪文件）
function Get-CsvNames([string]$dir) {
    return @(Get-ChildItem -LiteralPath $dir -Filter *.csv -File -ErrorAction SilentlyContinue |
        ForEach-Object { $_.Name })
}

$trainPath = Get-FullPath $TrainSet "训练集"
$testPath  = Get-FullPath $TestSet  "测试集"
$workPath  = if ([System.IO.Path]::IsPathRooted($WorkDir)) { $WorkDir } else { Join-Path $repo $WorkDir }
New-Item -ItemType Directory -Force -Path $workPath | Out-Null

$exePath   = Join-Path $workPath "eluose.exe"
$scoreCsv  = Join-Path $workPath "test_scores.csv"
$modelPath = Join-Path $workPath "q_weights.dat"

Write-Host "训练集：$trainPath"
Write-Host "测试集：$testPath"
Write-Host "批次：$BatchCount × $EpisodesPerBatch 局"
Write-Host ("epsilon：{0} → {1}，跨 {2} 批首尾相接" -f $EpsilonStart, $EpsilonEnd, $BatchCount)
Write-Host "工作目录：$workPath"

# 工作目录里已有上一次实验的 CSV：整体挪进 archive_<时间>，避免追加到旧表、覆盖旧追踪
$stale = @(Get-ChildItem -LiteralPath $workPath -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq "test_scores.csv" -or $_.Name -like "trace_batch*.csv" -or
                   $_.Name -like "model_batch*.dat" })
if ($stale.Count -gt 0) {
    $archive = Join-Path $workPath ("archive_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
    New-Item -ItemType Directory -Force -Path $archive | Out-Null
    $stale | ForEach-Object { Move-Item -LiteralPath $_.FullName -Destination $archive }
    Write-Host ("上一次实验的 {0} 个 CSV 已移到：{1}" -f $stale.Count, $archive)
}

# ---------- 1. 沿用已有模型：仓库里的 q_weights.dat 首次运行时复制过来 ----------
$repoModel = Join-Path $repo "q_weights.dat"
if (-not (Test-Path -LiteralPath $modelPath) -and (Test-Path -LiteralPath $repoModel)) {
    Copy-Item -LiteralPath $repoModel -Destination $modelPath
    Write-Host "已复制仓库模型作为起点：$repoModel"
}
if (-not (Test-Path -LiteralPath $modelPath)) {
    Write-Host "工作目录没有模型，将使用源码里的初始权重"
}

# ---------- 2. 循环：编译（按本批 epsilon 起点）→ 训练 → 测试 ----------
if (-not (Test-Path -LiteralPath $scoreCsv)) {
    [System.IO.File]::WriteAllText(
        $scoreCsv,
        "batch,episodes,cum_episodes,epsilon_start,epsilon_end,test_score,test_pieces,test_ms`r`n",
        (New-Object System.Text.UTF8Encoding($false)))
}

$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$totalEpisodes = $BatchCount * $EpisodesPerBatch
$cumEpisodes = 0

# 全局 epsilon 曲线：第 k 局训练（k 从 0 数起）用的 epsilon
function Get-ScheduleEpsilon([double]$episodeIndex) {
    if ($totalEpisodes -le 1) { return $EpsilonEnd }
    $frac = $episodeIndex / $totalEpisodes
    return $EpsilonStart + ($EpsilonEnd - $EpsilonStart) * $frac
}

for ($batch = 1; $batch -le $BatchCount; $batch++) {
    $label = "第 {0}/{1} 批" -f $batch, $BatchCount
    # 本批覆盖全局曲线的 [cumEpisodes, cumEpisodes + EpisodesPerBatch] 这一段
    $batchStart = Get-ScheduleEpsilon $cumEpisodes
    $batchEnd   = Get-ScheduleEpsilon ($cumEpisodes + $EpisodesPerBatch)
    $epsText    = $batchStart.ToString("0.000", $invariant)
    $epsEndText = $batchEnd.ToString("0.000", $invariant)
    Write-Host "`n$label：编译（epsilon $epsText → $epsEndText）→ 训练 $EpisodesPerBatch 局……"

    $build = Invoke-Captured $gcc @(
        "-std=c99", "-O2",
        "-DTRAIN_EPISODES=$EpisodesPerBatch",
        "-DTRAIN_EPSILON_START=$epsText",
        "-DTRAIN_EPSILON_END=$epsEndText",
        (Join-Path $repo "eluose.c"), "-o", $exePath, "-lm")
    if ($build.ExitCode -ne 0) {
        throw "$label 编译失败（gcc exit $($build.ExitCode)）：`n$($build.Output)"
    }

    $before = Get-CsvNames $workPath
    Push-Location -LiteralPath $workPath
    try {
        $run = Invoke-Captured $exePath @("--train", "-i", $trainPath)
    } finally {
        Pop-Location
    }
    $trainOut = $run.Output
    if ($run.ExitCode -ne 0) { throw "$label 训练失败（exit $($run.ExitCode)）：`n$trainOut" }

    $done = 0
    if ($trainOut -match "训练完成：(\d+) 局") { $done = [int]$Matches[1] }
    if ($done -le 0) { throw "$label 没有完成任何训练局：`n$trainOut" }
    $cumEpisodes += $done

    # 这一批新生成的追踪 CSV 就是本次训练的追踪
    $fresh = @(Get-CsvNames $workPath | Where-Object {
        $_ -ne "test_scores.csv" -and $before -notcontains $_
    })
    $savedTrace = Join-Path $workPath ("trace_batch{0:d3}.csv" -f $batch)
    if ($fresh.Count -ge 1) {
        $sourceTrace = Join-Path $workPath $fresh[0]
        Move-Item -LiteralPath $sourceTrace -Destination $savedTrace -Force
    } else {
        Write-Warning "$label 没有生成训练追踪文件（配置区 TRAINING_TRACE_PATH 可能是空串）"
    }

    # 测试时用的就是这一批训练完的模型，留一份快照便于逐批复盘
    $snapshot = Join-Path $workPath ("model_batch{0:d3}.dat" -f $batch)
    Copy-Item -LiteralPath $modelPath -Destination $snapshot -Force

    Write-Host "$label：测试集跑一遍……"
    Push-Location -LiteralPath $workPath
    try {
        $run = Invoke-Captured $exePath @("-i", $testPath)
    } finally {
        Pop-Location
    }
    $testOut = $run.Output
    if ($run.ExitCode -ne 0) { throw "$label 测试失败（exit $($run.ExitCode)）：`n$testOut" }

    # 对局每一步打印 "rotation x" 和当前累计得分；得分行是纯整数
    $lines = $testOut -split "\r?\n"
    $scoreLines = @($lines | Where-Object { $_ -match '^\d+$' })
    if ($scoreLines.Count -eq 0) { throw "$label 没能从测试输出里读到得分：`n$testOut" }
    $testScore = [int]$scoreLines[-1]
    $testPieces = $scoreLines.Count

    $testMs = 0
    if ($testOut -match "耗时：(\d+) ms") { $testMs = [int]$Matches[1] }

    $line = "{0},{1},{2},{3},{4},{5},{6},{7}" -f $batch, $done, $cumEpisodes, $epsText, $epsEndText, $testScore, $testPieces, $testMs
    [System.IO.File]::AppendAllText(
        $scoreCsv, $line + "`r`n", (New-Object System.Text.UTF8Encoding($false)))

    Write-Host ("{0}：epsilon {1} → {2}，训练 {3} 局（累计 {4}），测试得分 {5}，落子 {6} 步，用时 {7} ms" -f
        $label, $epsText, $epsEndText, $done, $cumEpisodes, $testScore, $testPieces, $testMs)
}

Write-Host "`n全部完成。结果："
Write-Host ("  测试得分表：{0}" -f $scoreCsv)
Write-Host ("  训练追踪：  {0}\trace_batch*.csv" -f $workPath)
Write-Host ("  模型快照：  {0}\model_batch*.dat" -f $workPath)
Write-Host ("  模型：      {0}" -f $modelPath)
Write-Host "`n出图命令："
Write-Host ("  python .\analyze_training.py `"{0}\trace_batch*.csv`" --test-log `"{1}`" --output `"{2}`"" -f
    $workPath, $scoreCsv, (Join-Path $workPath "training_analysis.png"))
