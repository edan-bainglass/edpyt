gcc -L. -Wall -Wno-unused-variable -Wl,-rpath=. -o $1 tests/$1.c -lfit -lm
