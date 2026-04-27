$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$workspaceRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $workspaceRoot

function Get-BootstrapPython {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{
            Executable = "python"
            Args = @()
        }
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @{
            Executable = "py"
            Args = @("-3.14")
        }
    }

    throw "Python 3.14 or newer is required to bootstrap this project."
}

$bootstrapPython = Get-BootstrapPython
$venvPython = Join-Path $workspaceRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment..."
    & $bootstrapPython.Executable @($bootstrapPython.Args + @("-m", "venv", ".venv"))
}

Write-Host "Installing project in editable mode..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e .

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

Write-Host "Verifying package import..."
& $venvPython -c "import autonomous_assistant; print('bootstrap ok')"

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Run the assistant with:"
Write-Host "  .\.venv\Scripts\autonomous-assistant run `"Summarize this repository.`""
