#include<stdio.h>
#include<stdlib.h>

int main() {

typedef struct {
    char    a;
    int    b;
    char    c;
}   Struct;

typedef struct {
    char    a;
    char    c;
    int    b;
}   Struct2;


Struct i;
char d='d';
printf("size of struct %d \n", sizeof(i));

return 0;
}

