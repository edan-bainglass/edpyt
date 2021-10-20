gcc -c -Wall -Wno-unused-variable -fpic [a-z]*.c
gcc -Wall -Wno-unused-variable -shared -o libfit.so [a-z]*.o -lm
