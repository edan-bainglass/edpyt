gcc -c -funroll-loops -O3 -Wall -Wno-unused-variable -fpic [a-z]*.c
gcc -Wall -Wno-unused-variable -shared -o libfit.so [a-z]*.o -lm -llapack -lblas
