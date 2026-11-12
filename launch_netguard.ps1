# Lance NetGuard IDS (interface du projet) puis ouvre le navigateur.
# Ne démarre le serveur que s'il ne tourne pas déjà.
param([switch]$NoBrowser)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py   = Join-Path $root ".venv\Scripts\python.exe"
$port = 8511

if (-not (Test-Path $py)) {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show("Environnement Python introuvable :`n$py", "NetGuard IDS") | Out-Null
    exit 1
}

$serving = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
if (-not $serving) {
    Start-Process -FilePath $py -ArgumentList "-m", "streamlit", "run", "app.py", "--server.port", "$port" `
        -WorkingDirectory $root -WindowStyle Minimized
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) { break }
    }
}

if (-not $NoBrowser) { Start-Process "http://localhost:$port" }
