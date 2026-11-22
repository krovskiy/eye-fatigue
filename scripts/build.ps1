param([switch]$Release)
$ErrorActionPreference = 'Stop'
$project = Join-Path $PSScriptRoot '..\app\EyeFatigue.csproj'
$config = if ($Release) { 'Release' } else { 'Debug' }
dotnet build $project -c $config --nologo
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
