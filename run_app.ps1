$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    py -m venv (Join-Path $ProjectRoot ".venv")
}

& $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

$ApiArgs = @("-m", "uvicorn", "project.main:app", "--host", "127.0.0.1", "--port", "8000")
Start-Process -FilePath $Python -ArgumentList $ApiArgs -WorkingDirectory $ProjectRoot -WindowStyle Hidden

& $Python -m streamlit run (Join-Path $ProjectRoot "project\app.py") --server.port 8501 --server.address 127.0.0.1
