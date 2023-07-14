FCI = ifort
FIFLAG = -align array64byte -xAVX2 -qopt-report4 -qopenmp -O2

  intel :
		$(FCI) $(FIFLAG) ./feb.f90 -o feb.xx
		
  clean:
		rm *.optrpt *.xx

 
