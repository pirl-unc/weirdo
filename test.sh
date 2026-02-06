#!/bin/bash
# Run tests with coverage
set -e

# Ensure weirdo is importable
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

pytest \
    --cov=weirdo/ \
    --cov-report=term-missing \
    test/
