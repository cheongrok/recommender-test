#!/bin/bash
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.6.1 python3 -
fi

export PATH="/opt/conda/bin:$HOME/.local/bin:$PATH"

cd /home/jovyan/work

echo "Configuring Poetry and exporting dependencies..."
poetry config virtualenvs.create false
poetry export --only=main,ml -f requirements.txt >> requirements.txt

echo "Installing dependencies from requirements.txt..."
/opt/conda/bin/pip install --no-cache-dir -r requirements.txt

echo "Cleaning up requirements.txt..."
rm requirements.txt

echo "Dependencies installed and cleaned up successfully."
