$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m notebook notebooks\fuel_consumption.ipynb --ServerApp.ip=127.0.0.1 --ServerApp.port=8890 --ServerApp.port_retries=20
