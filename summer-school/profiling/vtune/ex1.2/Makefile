
CIC = icc
CIFLAG = -xAVX2 -g -qopenmp -qopt-report4 -std=c99 -O2
  intel :
		$(CIC) $(CIFLAG) ./datatype_d.c -o dbl.xx
		$(CIC) $(CIFLAG) ./datatype_i.c -o int.xx
		$(CIC) $(CIFLAG) ./datatype_f.c -o flt.xx


  clean:
		rm *.S *.optrpt *.xx

  all : intel
