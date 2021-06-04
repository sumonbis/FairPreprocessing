#!/bin/sh
if [ $# -eq 3 ]
then
  cd $1/
  echo "Started running pipeline $2 for the task $1 for $3 times"
  for (( i=1;i<=$3;i++ ))
  do
    echo "Running $i th time."
    python $2.py
  done
  cd ..
else
  if [ $# -eq 2 ]
  then

    if [ $1 == 'german' ]
    then
       echo "Started running all the pipelines for german dataset $2 times."

       cd $1/
       for (( i=1;i<=$2;i++ ))
       do
         echo "Running $i th time."
         python GC1.py
         python GC2.py
         python GC3.py
         python GC4.py
         python GC5.py
         python GC6.py
         python GC7.py
         python GC8.py
         python GC9.py
         python GC10.py
       done
       cd ..
    fi

    if [ $1 == 'adult' ]
    then
       echo "Started running all the pipelines for adult dataset $2 times."

       cd $1/
       for (( i=1;i<=$2;i++ ))
       do
         python AC1.py
         python AC2.py
         python AC3.py
         python AC4.py
         python AC5.py
         python AC6.py
         python AC7.py
         python AC8.py
         python AC9.py
         python AC10.py
       done
       cd ..
    fi

    if [ $1 == 'bank' ]
    then
       echo "Started running all the pipelines for bank marketing dataset $2 times."

       cd $1/
       for (( i=1;i<=$2;i++ ))
       do
         python BM1.py
         python BM2.py
         python BM3.py
         python BM4.py
         python BM5.py
         python BM6.py
         python BM7.py
         python BM8.py
       done
       cd ..
    fi
    if [ $1 == 'titanic' ]
    then
       echo "Started running all the pipelines for titanic dataset $2 times."

       cd $1/
       for (( i=1;i<=$2;i++ ))
       do
         python TT1.py
         python TT2.py
         python TT3.py
         python TT4.py
         python TT5.py
         python TT6.py
         python TT7.py
         python TT8.py
       done
       cd ..
    fi
    if [ $1 == 'compas' ]
    then
       echo "Started running all the pipelines for compas dataset $2 times."

       cd $1/
       for (( i=1;i<=$2;i++ ))
       do
         python CP1.py
       done
       cd ..
    fi

  fi

fi
