#!/bin/bash
set -o errexit

ruff check weirdo/ test/

echo 'Passes ruff check'
