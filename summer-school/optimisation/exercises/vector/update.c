// This loop wont vectorize
// Make it vectorize by assuming each value in d is distinct
void update (float *a,float *b, float *d, int N, float s)
{
   for (int i = 0; i < N; i++) {
     int j = d[i];
     a[j] += s * b[i];
    }
}
