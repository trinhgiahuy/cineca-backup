#include <stdio.h>
#include <unistd.h>

int main (int argc, char** argv) {
  FILE *fp;
  float buff;
  int microbuff;
  
  fp = fopen(argv[1],"r");
  fscanf(fp,"%f",&buff);
  fclose(fp);
  
  printf("Waiting for %.3f seconds...\n",buff);
  microbuff=buff*1000000;
  usleep(microbuff);
  
  return 0;
}
