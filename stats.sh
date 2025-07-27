#!/bin/bash

echo "Model:"$1

export model=$1
calc "round($(cat *$model*|grep 'GOOD'|wc -l)/$(cat *$model*|wc -l),2)"
