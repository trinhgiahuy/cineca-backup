void mixed(float * a, double * b, float *c)
{
   for(int i = 1; i < 1000; ++i)
   {
      b[i] = b[i] - c[i];
      a[i] = a[i] + b[i];
   }
}
