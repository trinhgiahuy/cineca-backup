### Q- What is the difference between the two solutions proposed for C?

**A-** In solution A the parallel sections are opened inside the main, while in solution B they are restricted inside the 
functions. This makes the program easier to read and to modify for future improvements, since the wider a parallel section is, 
the most dangerous it can become when dealing with thread-safety.

Another change made in solution B is in the "evolve" function. Inside the first for loop, the temporary variables are declared 
inside the loop instead of outside, as in solution A: 

```
 #pragma omp parallel for
   for(int iy=1;iy<=NLY;++iy)
     for(int ix=1;ix<=NX;++ix){
       float temp0 = temp[((NX+2)*iy)+ix];
       temp_new[((NX+2)*iy)+ix] = temp0-(0.5*dt*(temp[((NX+2)*(iy+1))+ix]-temp[((NX+2)*(iy-1))+ix]+
       temp[((NX+2)*iy)+(ix+1)]-temp[((NX+2)*iy)+(ix-1)]))/dx;
     }
```
This may solve a thread-safety issue, since, if declared outside, the "temp0" variable is considered shared while it should be private
(of course, another solution is to declare it as private in the "parallel for" pragma).

Notice that the problem wouldn't sussist if the parallel section began inside the main, as in solution A 
(in such case the temp0 variable is already declared inside the parallel section).