#!/bin/bash
rsync -avcz --exclude .venv --delete guimauve:~/test-llm2/ .

