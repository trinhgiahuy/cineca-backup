CC = gcc
FC = pgf90
NVCC = nvcc

CFLAG = -O3
FFLAG = -Mcuda=cc3x -fast
NVCCFLAG = -arch=sm_37 -Xcompiler=-O3

dummy:
	@echo -e "make cuda\t\tcompile all CUDA sources"
	@echo -e "make cudafor\t\tcompile all CUDAFORTRAN sources"
	@echo -e "make clean\t\trm -f *.x *.mod"

cuda:    MMGlobCT_c.x
cudafor: MMGlobCT_f.x

MMGlobCT_c.x: MMGlobCT.cu
	$(NVCC) $(NVCCFLAG) $< -o $@

MMGlobCT_f.x: MMGlobCT.F90
	$(FC) $(FFLAG) $< -o $@

clean:
	rm -f *.x *.mod *.ptx
