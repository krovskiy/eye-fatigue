$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'build.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$exe = Join-Path $PSScriptRoot '..\app\bin\Debug\net10.0-windows\EyeFatigue.exe'
$destination = Join-Path $PSScriptRoot '..\screenshots'
$p = Start-Process -FilePath $exe -ArgumentList @('--screenshots', ('"' + $destination + '"')) -PassThru -Wait -WindowStyle Hidden
if ($p.ExitCode -ne 0) { exit $p.ExitCode }
Write-Host 'screenshots saved with sample activity'
