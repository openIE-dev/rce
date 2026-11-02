module kernels
  implicit none
contains
  pure function saxpy(a, x, y) result(z)
    real, intent(in) :: a
    real, intent(in) :: x(:), y(:)
    real :: z(size(x))
    integer :: i
    do concurrent (i = 1:size(x))
      z(i) = a * x(i) + y(i)
    end do
  end function saxpy
end module kernels

program main
  use kernels
  implicit none
  real :: x(8), y(8)
  integer :: i
  do i = 1, 8
    x(i) = real(i)
    y(i) = real(9 - i)
  end do
  print '(8f6.1)', saxpy(2.0, x, y)
end program main
