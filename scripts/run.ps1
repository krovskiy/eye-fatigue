$ErrorActionPreference = 'Stop'
$project = Join-Path $PSScriptRoot '..\app\EyeFatigue.csproj'
dotnet run --project $project
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
