! Standalone interface to unchanged speclib.f, built only in pytest tmp_path.
! REAL is promoted to 8 bytes, matching the explicitly selected reference build.
program reference_gll
  implicit none
  integer :: n, i, j
  real :: z(12), w(12), d(12,12), dt(12,12)
  read(*,*) n
  if (n < 2 .or. n > 12) stop 2
  call zwgll(z,w,n)
  call dgll(d,dt,z,n,12)
  do i=1,n
    write(*,'(*(ES25.16E3,1X))') z(i),w(i),(d(i,j),j=1,n)
  enddo
end program

! Error-only dependency of the linked spectral routines. Reaching it fails.
subroutine exitt
  stop 3
end subroutine
