#!/bin/bash

# to recreate venv
#uv venv
#uv pip install openai
#uv python install 3.12.11
#uv python pin 3.12.11

. .venv/bin/activate

uv pip list

python3.12 --version

python test2.py $1

