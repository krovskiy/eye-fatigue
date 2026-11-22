$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'build.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$exe = Join-Path $PSScriptRoot '..\app\bin\Debug\net10.0-windows\EyeFatigue.exe'
$p = Start-Process -FilePath $exe -ArgumentList '--self-test' -PassThru -Wait -WindowStyle Hidden
if ($p.ExitCode -ne 0) { exit $p.ExitCode }
Write-Host 'clock checks passed'
