<#
.SYNOPSIS
    Arranca el servidor de Voicebox (sin la app de escritorio) en http://127.0.0.1:17493

.DESCRIPTION
    Voicebox vive en su propio checkout con su propio .venv (PyTorch + CUDA), separado del
    .venv de Aeon para que sus versiones no choquen. Su carpeta se lee de VOICEBOX_DIR en .env
    (relativa al proyecto o absoluta); si no está, se usa .\voicebox dentro del proyecto.
    Hay que lanzarlo desde la raíz de Voicebox porque el backend usa imports relativos (backend.main:app).

.EXAMPLE
    .\scripts\start_voicebox.ps1
    Luego abre http://127.0.0.1:17493/docs para ver la API.
#>
param([int]$Port = 17493)

$root = Split-Path $PSScriptRoot -Parent

# VOICEBOX_DIR de .env (la última línea gana, como en pydantic-settings)
$envFile = Join-Path $root ".env"
$line = if (Test-Path $envFile) {
    Get-Content -Encoding utf8 $envFile | Where-Object { $_ -match '^\s*VOICEBOX_DIR\s*=' } | Select-Object -Last 1
}
$dir = if ($line) { ($line -split '=', 2)[1].Trim().Trim('"') } else { "voicebox" }
if (-not [IO.Path]::IsPathRooted($dir)) { $dir = Join-Path $root $dir }
$vb = [IO.Path]::GetFullPath($dir)
$py = Join-Path $vb ".venv\Scripts\python.exe"

if (-not (Test-Path (Join-Path $vb "backend\main.py"))) {
    throw "No encuentro Voicebox en $vb. Clónalo (git clone --branch v0.5.0 https://github.com/jamiepine/voicebox.git) y pon su carpeta en VOICEBOX_DIR dentro de .env"
}
if (-not (Test-Path $py)) {
    throw "No existe $py. Crea el entorno de Voicebox siguiendo su README."
}

# Si ya hay un Voicebox respondiendo no lanzamos otro: chocaría por el puerto
try {
    $health = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "Voicebox ya está corriendo en http://127.0.0.1:$Port ($($health.status))." -ForegroundColor Green
    return
}
catch {}

Push-Location $vb
try {
    & $py -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
}
finally {
    Pop-Location
}
