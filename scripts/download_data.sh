#!/usr/bin/env bash
set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

HTTPX_DIR="$DATA_DIR/httpx_src"

if [ -d "$HTTPX_DIR" ]; then
    echo "httpx source already exists at $HTTPX_DIR"
    exit 0
fi

echo "Fetching httpx source location from pip..."

HTTPX_PKG=$(python3 -c "import httpx, pathlib; print(pathlib.Path(httpx.__file__).parent)" 2>/dev/null || true)

if [ -z "$HTTPX_PKG" ]; then
    echo "httpx not installed. Installing..."
    pip3 install httpx
    HTTPX_PKG=$(python3 -c "import httpx, pathlib; print(pathlib.Path(httpx.__file__).parent)")
fi

echo "Copying httpx source from $HTTPX_PKG ..."
cp -r "$HTTPX_PKG" "$HTTPX_DIR"

python3 -c "
import pathlib
src = pathlib.Path('$HTTPX_DIR')
files = list(src.rglob('*.py'))
chars = sum(len(f.read_text(errors='replace')) for f in files)
print(f'Files : {len(files)}')
print(f'Tokens: ~{chars//4:,}')
"