#!/bin/sh
if [ $# -eq 2 ]
then
  cd $1/
  echo "Started running pipeline $2 for the task $1"
  python $2.py
  cd ..
fi
 
if [ $# -eq 1 ]
then
  if [ $1 == 'all' ]
  then
     echo "Started running all the pipeline for all the tasks."

     cd german/
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
     cd ..

     cd adult/
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
     cd ..

     cd bank/
     python BM1.py
     python BM2.py
     python BM3.py
     python BM4.py
     python BM5.py
     python BM6.py
     python BM7.py
     python BM8.py
     cd ..

     cd compas/
     python CP1.py
     cd ..

     cd titanic/
     python TT1.py
     python TT2.py
     python TT3.py
     python TT4.py
     python TT5.py
     python TT6.py
     python TT7.py
     python TT8.py
     cd ..
  fi

  if [ $1 == 'german' ]
  then
     echo "Started running all the pipelines for german dataset."

     cd $1/
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
     cd ..
  fi

  if [ $1 == 'adult' ]
  then
     echo "Started running all the pipelines for adult dataset."

     cd $1/
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
     cd ..
  fi

  if [ $1 == 'bank' ]
  then
     echo "Started running all the pipelines for bank marketing dataset."

     cd $1/
     python BM1.py
     python BM2.py
     python BM3.py
     python BM4.py
     python BM5.py
     python BM6.py
     python BM7.py
     python BM8.py
     cd ..
  fi
  if [ $1 == 'titanic' ]
  then
     echo "Started running all the pipelines for titanic dataset."

     cd $1/
     python TT1.py
     python TT2.py
     python TT3.py
     python TT4.py
     python TT5.py
     python TT6.py
     python TT7.py
     python TT8.py
     cd ..
  fi
  if [ $1 == 'compas' ]
  then
     echo "Started running all the pipelines for compas dataset."

     cd $1/
     python CP1.py
     cd ..
  fi

fi
