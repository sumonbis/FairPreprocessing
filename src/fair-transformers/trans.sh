#!/bin/sh
if [ $# -eq 2 ]
then
  echo "Started running transformers for the task $1 for $2 times"
  for (( i=1;i<=$2;i++ ))
  do
    echo "Running $i th time."
    python $1.py
  done
fi
