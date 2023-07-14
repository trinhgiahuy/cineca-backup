echo "MM no avx"
./matrixNoVec.x 
echo "MM AVX"
./matrixAVX.x 
echo "DGEMM"
./matrixAVXdGEMM.x 
echo "SGEMM"
./matrixAVXsGEMM.x