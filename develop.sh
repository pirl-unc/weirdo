#!/bin/bash
# Development install script
# Activates venv if needed and installs package in editable mode
set -e

# Check if we're already in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "No virtual environment active, looking for one..."

    # Common venv locations to check
    VENV_PATHS=(
        ".venv"
        "venv"
        "../shared-virtual-env"
        "$HOME/.virtualenvs/weirdo"
    )

    FOUND_VENV=""
    for venv_path in "${VENV_PATHS[@]}"; do
        if [[ -f "$venv_path/bin/activate" ]]; then
            FOUND_VENV="$venv_path"
            break
        fi
    done

    if [[ -n "$FOUND_VENV" ]]; then
        echo "Found venv at: $FOUND_VENV"
        source "$FOUND_VENV/bin/activate"
    else
        echo "No venv found. Creating one with uv..."
        if command -v uv &> /dev/null; then
            uv venv .venv
            source .venv/bin/activate
        else
            echo "uv not found, using python -m venv"
            python3 -m venv .venv
            source .venv/bin/activate
        fi
    fi

    echo "Activated: $VIRTUAL_ENV"
fi

# Install in editable mode (prefer uv if available)
echo "Installing weirdo in editable mode..."
if command -v uv &> /dev/null; then
    uv pip install -e ".[dev]"
else
    pip install -e ".[dev]"
fi

echo "Done! weirdo installed in development mode."
