#!/bin/bash
# Deploy to PyPI
# Prerequisites:
#   - Clean working directory (all changes committed)
#   - PyPI credentials configured (~/.pypirc or TWINE_* env vars)
set -e

echo "Running lint..."
./lint.sh

echo "Running tests..."
./test.sh

echo "Installing build tools..."
if command -v uv >/dev/null 2>&1; then
  echo "Using uv for build tooling..."
  if [ ! -d ".venv" ]; then
    uv venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install --upgrade build twine
  BUILD_CMD="python -m build"
  TWINE_CMD="python -m twine"
else
  python3 -m pip install --upgrade build twine
  BUILD_CMD="python3 -m build"
  TWINE_CMD="python3 -m twine"
fi

echo "Cleaning old builds..."
rm -rf dist/ build/ *.egg-info

echo "Building package..."
$BUILD_CMD

echo "Checking git status..."
git --version

echo "Uploading to PyPI..."
$TWINE_CMD upload --skip-existing dist/*

echo "Creating git tag..."
VERSION=$(python3 - <<'PY'
import re
from pathlib import Path
text = Path("weirdo/__init__.py").read_text()
match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
print(match.group(1))
PY
)
git tag "v${VERSION}"

echo "Pushing tags..."
git push --tags

echo "Deployed version ${VERSION} successfully!"
