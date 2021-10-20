gcc -L. -Wl,-rpath=. -o $1 tests/$1.c -lfit -lm
