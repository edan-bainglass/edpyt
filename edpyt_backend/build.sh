gcc -c -Wall -Wno-unused-variable -fpic [a-z]*.c
gcc -shared -o libfit.so [a-z]*.o -lm
gcc -L. -Wl,-rpath=. -o main main.c -lfit
./main
python fit.py && python test_fit.py
