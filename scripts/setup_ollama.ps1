<#
.SYNOPSIS
    BoardMind AI - Ollama Local LLM Setup Script
    Installs Ollama and pulls recommended models for 100% offline AI concept synthesis.

.DESCRIPTION
    This script:
    1. Checks if Ollama is installed
    2. Installs Ollama if missing (Windows)
    3. Starts Ollama service
    4. Pulls recommended models: llama3.2:3b (primary), qwen2.5:3b (fallback)
    5. Verifies the installation

.NOTES
    Requires Administrator privileges for installation.
    Run from project root: .\scripts\setup_ollama.ps1
#>

param(
    [switch]$SkipInstall,
    [switch]$SkipModelPull,
    [string]$OllamaHost = "http://127.0.0.1:11434",
    [string[]]$Models = @("llama3.2:3b", "qwen2.5:3b")
)

$ErrorActionPreference = "Stop"

function Write-Header {
    param([string]$Message)
    Write-Host "`n============================================" -ForegroundColor Cyan
    Write-Host "  $Message" -ForegroundColor Cyan
    Write-Host "============================================`n" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-WarningMsg {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor White
}

function Check-AdminRights {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]$currentUser
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Is-OllamaInstalled {
    try {
        $version = ollama --version 2>$null
        return $true
    } catch {
        return $false
    }
}

function Install-Ollama {
    Write-Header "Installing Ollama"

    if (-not (Check-AdminRights)) {
        Write-ErrorMsg "Administrator privileges required for Ollama installation."
        Write-Info "Please run PowerShell as Administrator and re-run this script."
        exit 1
    }

    Write-Info "Downloading Ollama installer..."
    $installerPath = "$env:TEMP\OllamaSetup.exe"

    try {
        Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $installerPath -UseBasicParsing
        Write-Success "Downloaded installer to $installerPath"
    } catch {
        Write-ErrorMsg "Failed to download Ollama installer: $_"
        exit 1
    }

    Write-Info "Running Ollama installer (silent)..."
    try {
        $process = Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait -PassThru
        if ($process.ExitCode -ne 0) {
            Write-ErrorMsg "Ollama installer exited with code $($process.ExitCode)"
            exit 1
        }
        Write-Success "Ollama installed successfully"
    } catch {
        Write-ErrorMsg "Installation failed: $_"
        exit 1
    } finally {
        if (Test-Path $installerPath) { Remove-Item $installerPath -Force }
    }

    # Add to PATH for current session
    $ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama"
    if (Test-Path $ollamaPath) {
        $env:PATH += ";$ollamaPath"
        Write-Info "Added $ollamaPath to PATH for this session"
    }
}

function Start-OllamaService {
    Write-Header "Starting Ollama Service"

    # Check if already running
    try {
        $response = Invoke-RestMethod -Uri "$OllamaHost/api/tags" -Method Get -TimeoutSec 2 -ErrorAction Stop
        Write-Success "Ollama service already running at $OllamaHost"
        return
    } catch {
        Write-Info "Ollama service not responding, attempting to start..."
    }

    # Start ollama serve in background
    $ollamaExe = "ollama"
    if (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe") {
        $ollamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    }

    Write-Info "Starting ollama serve..."
    $process = Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden -PassThru

    # Wait for service to be ready
    Write-Info "Waiting for Ollama service to be ready..."
    $maxAttempts = 30
    $attempt = 0

    while ($attempt -lt $maxAttempts) {
        Start-Sleep -Seconds 1
        $attempt++
        try {
            $response = Invoke-RestMethod -Uri "$OllamaHost/api/tags" -Method Get -TimeoutSec 2 -ErrorAction Stop
            Write-Success "Ollama service started successfully at $OllamaHost"
            return
        } catch {
            if ($attempt % 5 -eq 0) {
                Write-Info "Still waiting... ($attempt/$maxAttempts)"
            }
        }
    }

    Write-ErrorMsg "Ollama service failed to start within timeout"
    exit 1
}

function Pull-OllamaModels {
    param([string[]]$Models)

    Write-Header "Pulling Ollama Models"

    foreach ($model in $Models) {
        Write-Info "Checking if model '$model' exists..."
        try {
            $response = Invoke-RestMethod -Uri "$OllamaHost/api/tags" -Method Get -TimeoutSec 5
            $exists = $response.models | Where-Object { $_.name -eq $model -or $_.name -like "$model:*" }

            if ($exists) {
                Write-Success "Model '$model' already available"
                continue
            }
        } catch {
            Write-WarningMsg "Could not check existing models: $_"
        }

        Write-Info "Pulling model '$model' (this may take several minutes)..."
        try {
            $pullProcess = Start-Process -FilePath "ollama" -ArgumentList "pull $model" -Wait -PassThru
            if ($pullProcess.ExitCode -eq 0) {
                Write-Success "Model '$model' pulled successfully"
            } else {
                Write-ErrorMsg "Failed to pull model '$model' (exit code: $($pullProcess.ExitCode))"
            }
        } catch {
            Write-ErrorMsg "Error pulling model '$model': $_"
        }
    }
}

function Verify-Installation {
    param([string[]]$Models)

    Write-Header "Verifying Installation"

    # Check service
    try {
        $response = Invoke-RestMethod -Uri "$OllamaHost/api/tags" -Method Get -TimeoutSec 5
        Write-Success "Ollama API responding at $OllamaHost"

        $availableModels = $response.models | Select-Object -ExpandProperty name
        Write-Info "Available models: $($availableModels -join ', ')"

        foreach ($model in $Models) {
            $matched = $availableModels | Where-Object { $_ -eq $model -or $_ -like "$model:*" }
            if ($matched) {
                Write-Success "Required model '$model' is available"
            } else {
                Write-WarningMsg "Required model '$model' NOT found"
            }
        }
    } catch {
        Write-ErrorMsg "Ollama API not accessible: $_"
        exit 1
    }

    # Test generation
    Write-Info "Testing model generation with llama3.2:3b..."
    try {
        $testPayload = @{
            model = "llama3.2:3b"
            prompt = "Say 'BoardMind AI ready' in exactly 3 words."
            stream = $false
            options = @{ temperature = 0.1; num_predict = 10 }
        } | ConvertTo-Json -Depth 5

        $response = Invoke-RestMethod -Uri "$OllamaHost/api/generate" -Method Post -Body $testPayload -ContentType "application/json" -TimeoutSec 30
        Write-Success "Generation test passed: $($response.response.Trim())"
    } catch {
        Write-WarningMsg "Generation test failed (model may still be loading): $_"
    }
}

function Create-EnvFile {
    Write-Header "Creating .env Configuration"

    $envPath = "$PSScriptRoot\..\backend\.env"
    $envExamplePath = "$PSScriptRoot\..\backend\.env.example"

    if (Test-Path $envPath) {
        Write-Info ".env already exists at $envPath, skipping creation"
        return
    }

    if (Test-Path $envExamplePath) {
        Copy-Item $envExamplePath $envPath -Force
        Write-Success "Created .env from .env.example at $envPath"
    } else {
        $envContent = @"
# BoardMind AI environment configuration
# Local Offline LLM via Ollama (Recommended for 100% Private Offline Viva Demo)
# Install from https://ollama.com and run: ollama run llama3.2:3b
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b

# Optional Cloud Fallbacks:
# Groq API key for fast Llama-3.3 70B (https://console.groq.com/keys)
GROQ_API_KEY=

# Optional Google Gemini Vision API key
GEMINI_API_KEY=
"@
        Set-Content -Path $envPath -Value $envContent -Encoding UTF8
        Write-Success "Created default .env at $envPath"
    }
}

# ==================== MAIN ====================
Write-Header "BoardMind AI - Ollama Local LLM Setup"
Write-Info "Project root: $PSScriptRoot\.."

# 1. Check/Install Ollama
if (-not $SkipInstall) {
    if (Is-OllamaInstalled) {
        Write-Success "Ollama already installed: $(ollama --version)"
    } else {
        Write-WarningMsg "Ollama not found"
        Install-Ollama
    }
} else {
    Write-Info "Skipping Ollama installation (--SkipInstall)"
}

# 2. Start Ollama service
Start-OllamaService

# 3. Pull models
if (-not $SkipModelPull) {
    Pull-OllamaModels -Models $Models
} else {
    Write-Info "Skipping model pull (--SkipModelPull)"
}

# 4. Verify
Verify-Installation -Models $Models

# 5. Create .env
Create-EnvFile

Write-Header "Setup Complete!"
Write-Success "Ollama is ready for BoardMind AI offline concept synthesis."
Write-Info "Next steps:"
Write-Info "  1. Install Python dependencies: pip install -r backend/requirements.txt"
Write-Info "  2. Start backend: cd backend && python -m uvicorn app.main:app --reload"
Write-Info "  3. Test API: curl -X POST http://localhost:8000/api/process -F 'file=@test_image.jpg'"