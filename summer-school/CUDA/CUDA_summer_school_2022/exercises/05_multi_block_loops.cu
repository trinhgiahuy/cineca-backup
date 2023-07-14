#include <stdio.h>

void doLoop(int N) {
  for (int i = 0; i < N; ++i)
    printf("%d\n", i);
}

int main()
{
    int N = 10;
    doLoop(N);
}
	