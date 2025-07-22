#!/bin/bash
rsync -av --exclude .venv --delete . guimauve:~/test-llm/

