#!/bin/bash
# BoardMind AI - Ollama Local LLM Setup Script (Linux/macOS)
# Installs Ollama and pulls recommended models for 100% offline AI concept synthesis.

set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
MODELS=("llama3.2:3b" "qwen2.5:3b")
SKIP_INSTALL=false
SKIP_MODEL_PULL=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --skip-install) SKIP_INSTALL=true ;;
        --skip-model-pull) SKIP_MODEL_PULL=true ;;
        --host=*) OLLAMA_HOST="${arg#*=}" ;;
        --models=*) IFS=',' read -ra MODELS <<< "${arg#*=}" ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_header() { echo -e "\n${CYAN}============================================${NC}\n${CYAN}  $1${NC}\n${CYAN}============================================${NC}\n"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_info() { echo -e "[INFO] $1"; }

is_ollama_installed() {
    command -v ollama >/dev/null 2>&1
}

install_ollama() {
    log_header "Installing Ollama"

    if is_ollama_installed; then
        log_success "Ollama already installed: $(ollama --version)"
        return
    fi

    log_info "Downloading and installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh

    if is_ollama_installed; then
        log_success "Ollama installed successfully"
    else
        log_error "Ollama installation failed"
        exit 1
    fi
}

start_ollama_service() {
    log_header "Starting Ollama Service"

    # Check if already running
    if curl -sf --max-time 2 "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
        log_success "Ollama service already running at $OLLAMA_HOST"
        return
    fi

    log_info "Starting ollama serve in background..."
    ollama serve >/dev/null 2>&1 &
    OLLAMA_PID=$!

    # Wait for service to be ready
    log_info "Waiting for Ollama service to be ready..."
    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        sleep 1
        attempt=$((attempt + 1))
        if curl -sf --max-time 2 "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
            log_success "Ollama service started successfully at $OLLAMA_HOST"
            return
        fi
        if [ $((attempt % 5)) -eq 0 ]; then
            log_info "Still waiting... ($attempt/$max_attempts)"
        fi
    done

    log_error "Ollama service failed to start within timeout"
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
}

pull_models() {
    log_header "Pulling Ollama Models"

    for model in "${MODELS[@]}"; do
        log_info "Checking if model '$model' exists..."
        if ollama list | grep -q "^$model "; then
            log_success "Model '$model' already available"
            continue
        fi

        log_info "Pulling model '$model' (this may take several minutes)..."
        if ollama pull "$model"; then
            log_success "Model '$model' pulled successfully"
        else
            log_error "Failed to pull model '$model'"
        fi
    done
}

verify_installation() {
    log_header "Verifying Installation"

    # Check service
    if ! curl -sf --max-time 5 "$OLLAMA_HOST/api/tags" >/dev/null; then
        log_error "Ollama API not accessible at $OLLAMA_HOST"
        exit 1
    fi

    log_success "Ollama API responding at $OLLAMA_HOST"

    local available_models
    available_models=$(curl -sf "$OLLAMA_HOST/api/tags" | jq -r '.models[].name' 2>/dev/null || echo "")
    log_info "Available models: ${available_models:-none}"

    for model in "${MODELS[@]}"; do
        if echo "$available_models" | grep -q "^$model$"; then
            log_success "Required model '$model' is available"
        else
            log_warn "Required model '$model' NOT found"
        fi
    done

    # Test generation
    log_info "Testing model generation with llama3.2:3b..."
    local test_payload='{"model":"llama3.2:3b","prompt":"Say '\''BoardMind AI ready'\'' in exactly 3 words.","stream":false,"options":{"temperature":0.1,"num_predict":10}}'

    if response=$(curl -sf --max-time 30 -X POST "$OLLAMA_HOST/api/generate" -H "Content-Type: application/json" -d "$test_payload" 2>/dev/null); then
        local result=$(echo "$response" | jq -r '.response' 2>/dev/null | tr -d '\n')
        log_success "Generation test passed: $result"
    else
        log_warn "Generation test failed (model may still be loading)"
    fi
}

create_env_file() {
    log_header "Creating .env Configuration"

    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local env_path="$script_dir/../backend/.env"
    local env_example_path="$script_dir/../backend/.env.example"

    if [ -f "$env_path" ]; then
        log_info ".env already exists at $env_path, skipping creation"
        return
    fi

    if [ -f "$env_example_path" ]; then
        cp "$env_example_path" "$env_path"
        log_success "Created .env from .env.example at $env_path"
    else
        cat > "$env_path" << 'EOF'
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
EOF
        log_success "Created default .env at $env_path"
    fi
}

# ==================== MAIN ====================
log_header "BoardMind AI - Ollama Local LLM Setup"
log_info "Project root: $(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. Check/Install Ollama
if [ "$SKIP_INSTALL" = false ]; then
    install_ollama
else
    log_info "Skipping Ollama installation (--skip-install)"
fi

# 2. Start Ollama service
start_ollama_service

# 3. Pull models
if [ "$SKIP_MODEL_PULL" = false ]; then
    pull_models
else
    log_info "Skipping model pull (--skip-model-pull)"
fi

# 4. Verify
verify_installation

# 5. Create .env
create_env_file

log_header "Setup Complete!"
log_success "Ollama is ready for BoardMind AI offline concept synthesis."
log_info "Next steps:"
log_info "  1. Install Python dependencies: pip install -r backend/requirements.txt"
log_info "  2. Start backend: cd backend && python -m uvicorn app.main:app --reload"
log_info "  3. Test API: curl -X POST http://localhost:8000/api/process -F 'file=@test_image.jpg'"