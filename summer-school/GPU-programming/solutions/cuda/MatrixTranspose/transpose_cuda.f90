module matrix_transpose
  use cudafor
  implicit none

  integer, parameter :: tile_dim = 32
  integer, parameter :: block_dim = 32

contains

  attributes(global) subroutine copy(a, b, n, m, nreps)
    real, device  :: a(n, m), b(m, n)
    integer, value :: n, m, nreps
    integer :: ix, iy, r
    ix = (blockIdx%x-1) * blockDim%x + threadIdx%x
    iy = (blockIdx%y-1) * blockDim%y + threadIdx%y
    do r=1,nreps
      b(ix,iy) = a(ix,iy)
    enddo
  end subroutine copy

  attributes(global) subroutine transpose_naive(a, b, n, m, nreps)
    real, device  :: a(n, m), b(m, n)
    integer, value :: n, m, nreps
    integer :: ix, iy, r
    ix = (blockIdx%x-1) * blockDim%x + threadIdx%x
    iy = (blockIdx%y-1) * blockDim%y + threadIdx%y
    do r=1,nreps
      b(ix,iy) = a(iy,ix)
    enddo
  end subroutine transpose_naive

  attributes(global) subroutine transpose_coalesced(a, b, n, m, nreps)
    real, device  :: a(n, m), b(m, n)
    integer, value :: n, m, nreps
    integer :: ix, iy, jx, jy, kx, ky, r
    real, shared :: tile(tile_dim, tile_dim)

    ix = threadIdx%x
    iy = threadIdx%y

    do r=1,nreps
      jx = (blockIdx%x-1) * tile_dim + ix
      jy = (blockIdx%y-1) * tile_dim + iy

      kx = (blockIdx%y-1) * tile_dim + ix
      ky = (blockIdx%x-1) * tile_dim + iy

      tile(ix,iy) = a(jx,jy)
      call syncthreads()
      b(kx,ky) = tile(iy,ix)
    enddo  

  end subroutine transpose_coalesced

  attributes(global) subroutine transpose_nobank(a, b, n, m, nreps)
    real, device  :: a(n, m), b(m, n)
    integer, value :: n, m, nreps
    integer :: ix, iy, jx, jy, kx, ky, r
    real, shared :: tile(tile_dim+1,tile_dim)

    ix = threadIdx%x
    iy = threadIdx%y

    do r=1,nreps
      jx = (blockIdx%x-1) * tile_dim + ix
      jy = (blockIdx%y-1) * tile_dim + iy

      kx = (blockIdx%y-1) * tile_dim + ix
      ky = (blockIdx%x-1) * tile_dim + iy

      tile(ix,iy) = a(jx,jy)
      call syncthreads()
      b(kx,ky) = tile(iy,ix)
    enddo
  end subroutine transpose_nobank

  subroutine repeat_inner (ADEV, BDEV, N, M, NREPS, kernel, dimGrid, dimBlock, kernel_name)
    implicit none
    type(dim3) :: dimGrid, dimBlock
    integer, value :: N, M, NREPS
    real, device  :: ADEV(N, M), BDEV(M, N)
    character(*), intent(in) :: kernel_name
    type(cudaEvent) :: ev_start, ev_stop
    real :: inner_time
    real :: inner_bandwidth = 1.0
    integer err
    interface
      attributes(global) subroutine kernel(a, b, n, m, nreps)
        integer, value :: n, m, nreps
        real, device  :: a(n, m), b(m, n)
      end subroutine
    end interface
  

    err = cudaEventCreate(ev_start)
    err = cudaEventCreate(ev_stop)
    err = cudaEventRecord(ev_start, 0)
    call kernel<<<dimgrid, dimblock>>> (ADEV, BDEV, N, M, NREPS)
    err = cudaEventRecord(ev_stop, 0)
    err = cudaEventSynchronize(ev_stop)
    err = cudaEventElapsedTime(inner_time, ev_start, ev_stop)
    inner_time = inner_time/NREPS
    inner_bandwidth = (2.0 * 1000 * N*M*4) / (1024*1024*1024) / inner_time

    write(*,'("inner ",a16," Throughput",F12.4," GB/s, Time ",F9.4," s")')  kernel_name,inner_bandwidth,inner_time
    err = cudaEventDestroy(ev_start)
    err = cudaEventDestroy(ev_stop)
  end subroutine

  subroutine repeat_outer (ADEV, BDEV, N, M, NREPS, kernel, dimGrid, dimBlock, kernel_name)
    implicit none
    type(dim3) :: dimGrid, dimBlock
    integer, value :: N, M, NREPS
    real, device  :: ADEV(N, M), BDEV(M, N)
    character(*), intent(in) :: kernel_name
    type(cudaEvent) :: ev_start, ev_stop
    real :: outer_time
    real :: outer_bandwidth = 1.0
    integer err, r
    interface
      attributes(global) subroutine kernel(a, b, n, m, nreps)
        integer, value :: n, m, nreps
        real, device  :: a(n, m), b(m, n)
      end subroutine
    end interface
  
    err = cudaEventCreate(ev_start)
    err = cudaEventCreate(ev_stop)

    err = cudaEventRecord(ev_start, 0)
    do r=1,NREPS
      call kernel<<<dimgrid, dimblock>>> (ADEV, BDEV, N, M, 1)
    enddo
    err = cudaEventRecord(ev_stop, 0)
    err = cudaEventSynchronize(ev_stop)
    err = cudaEventElapsedTime(outer_time, ev_start, ev_stop)
    outer_time = outer_time/NREPS
    outer_bandwidth = (2.0 * 1000 * N*M*4) / (1024*1024*1024) / outer_time

    write(*,'("outer ",a16," Throughput",F12.4," GB/s, Time ",F9.4," s")')  kernel_name,outer_bandwidth,outer_time
    err = cudaEventDestroy(ev_start)
    err = cudaEventDestroy(ev_stop)
  end subroutine
end module matrix_transpose

program matrix_transpose_gpu
  use matrix_transpose
  use cudafor
  implicit none

  integer, parameter :: N = 448, M=448
  integer, parameter :: NREPS = 1000

  integer :: i,j
  type(dim3) :: dimGrid, dimBlock
  real, allocatable, dimension(:,:) :: A, B
  real, device, allocatable, dimension(:,:) :: ADEV, BDEV

  if (mod(N,tile_dim).ne.0 .or. mod(M,tile_dim).ne.0) then
    print *, 'Matrix size must be integral multiple of tile size'
    print *, 'Exiting...'
    stop
  endif

  ! print *, 'Matrix size: ',N,'x',M,' (',N/tile_dim,'x',M/tile_dim,' tiles)'
  ! print *, 'tile size: ',tile_dim,'x',tile_dim,', block size: ',tile_dim,'x',block_dim

  allocate(A(N,M))
  allocate(B(M,N))
  do j=1,M
    do i=1,N
      A(i,j) = (j-1)*N+i;
    enddo
  enddo
  ! print *, A(1,1), A(1,2), A(N,M)

  allocate(ADEV(N,M))
  allocate(BDEV(M,N))
  ADEV = A

  dimGrid  = dim3(M/tile_dim, N/tile_dim, 1)
  dimBlock = dim3(tile_dim, block_dim, 1)

  print *, ' '
  print *, 'Starting outer measure tests'
  call repeat_outer(ADEV, BDEV, N, M, NREPS, copy, dimGrid, dimBlock,'copy')
  call repeat_outer(ADEV, BDEV, N, M, NREPS, transpose_naive, dimGrid, dimBlock,'trans_naive')
  call repeat_outer(ADEV, BDEV, N, M, NREPS, transpose_coalesced, dimGrid, dimBlock,'trans_coalesced')
  call repeat_outer(ADEV, BDEV, N, M, NREPS, transpose_nobank, dimGrid, dimBlock,'trans_nobank')

  print *, ' '
  print *, 'Starting inner measure tests'
  call repeat_inner(ADEV, BDEV, N, M, NREPS, copy, dimGrid, dimBlock,'copy')
  call repeat_inner(ADEV, BDEV, N, M, NREPS, transpose_naive, dimGrid, dimBlock,'trans_naive')
  call repeat_inner(ADEV, BDEV, N, M, NREPS, transpose_coalesced, dimGrid, dimBlock,'trans_coalesced')
  call repeat_inner(ADEV, BDEV, N, M, NREPS, transpose_nobank, dimGrid, dimBlock,'trans_nobank')

  print *, ' '
  print *, 'PASSED'
  B = BDEV
  ! print *, B(1,1), B(2,1), B(M,N)
  deallocate(ADEV)
  deallocate(BDEV)
end program matrix_transpose_gpu
