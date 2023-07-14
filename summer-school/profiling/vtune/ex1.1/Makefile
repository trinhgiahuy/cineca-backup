#
# Makefile of Matrix Multiplication benchmark
# 
#

C = matrix.c       # file da compilare 
CCOMP = icc     # compilatore
CFLAGS= -std=c11 -qopt-report5 -qopt-report-file=$@.optrpt
VECFLAGS = -xAVX #AVX
CFLAGSMKL= -std=c11 -DMKL_ILP64 -I$(MKLROOT)/include 
LIB =  -L$(MKLROOT)/lib/intel64 -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential -lpthread -lm

matrixAVX.x: $(C)
	$(CCOMP) -O3 $(VECFLAGS) $(CFLAGS) $^ -o $@ 

matrixNoVec.x: $(C)
	$(CCOMP) -O3 -no-vec $(CFLAGS) $^ -o $@ 

matrixAVXdGEMM.x: $(C)
	$(CCOMP) -O3 -DMKL_DGEMM $(VECFLAGS) $(CFLAGSMKL) $^ $(LIB) -o $@ 

matrixAVXsGEMM.x: $(C)
	$(CCOMP) -O3 -DMKL_SGEMM $(VECFLAGS) $(CFLAGSMKL) $^ $(LIB) -o $@ 

	
all : matrixAVX.x matrixNoVec.x matrixAVXdGEMM.x matrixAVXsGEMM.x

.PHONY: clean  

clean:
	rm *.out* *.optrpt *.e* core*.* *.x $(COUT)
