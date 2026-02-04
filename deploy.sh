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
  uv tool install --quiet build twine
  BUILD_CMD="uv tool run build"
  TWINE_CMD="uv tool run twine"
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
$TWINE_CMD upload dist/*

echo "Creating git tag..."
VERSION=$(python3 -c "from weirdo import __version__; print(__version__)")
git tag "v${VERSION}"

echo "Pushing tags..."
git push --tags

echo "Deployed version ${VERSION} successfully!"
