CEXECUTABLE  := transpose_c.x
FEXECUTABLE  := transpose_f.x

CUFILES     := transpose.cu
FCUFILES    := transpose_cuda.f90
SDKDIR      := $(CUDADIR)/NVIDIA_CUDA_SDK/C

################################################################################
# Rules and targets

all: c fortran

c: $(CEXECUTABLE) 

fortran: $(FEXECUTABLE)

$(CEXECUTABLE): $(CUFILES)
	nvcc -arch=sm_37 -o $@ $<

$(FEXECUTABLE): $(FCUFILES)
	pgf90 -Mcuda=cc3x -o $@ $<

clean:
	rm -rf $(CEXECUTABLE) $(FEXECUTABLE) *.mod
cleanall: clean
