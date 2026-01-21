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
python3 -m pip install --upgrade build twine

echo "Cleaning old builds..."
rm -rf dist/ build/ *.egg-info

echo "Building package..."
python3 -m build

echo "Checking git status..."
git --version

echo "Uploading to PyPI..."
python3 -m twine upload dist/*

echo "Creating git tag..."
VERSION=$(python3 -c "from weirdo import __version__; print(__version__)")
git tag "v${VERSION}"

echo "Pushing tags..."
git push --tags

echo "Deployed version ${VERSION} successfully!"
