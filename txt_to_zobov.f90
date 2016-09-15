        program main
    
        character*100 infile, outfile, cj
        integer n, i, nn, nlines
        character*2  temp
!        parameter (n=439264)
        real*8, ALLOCATABLE :: x(:), y(:), z(:) 
!        real*8 x(n), y(n), z(n) 
        real*8 t1,t2 
 
            infile='KiDS_Mocks_gal_pos_0_525_cube_geo.txt'
           outfile= 'KiDS_Mocks_gal_pos_0_525_cube_geo.dat'                                     
     

         open(10, file=infile)
         read(10,*)n
         print*, n
         allocate(x(n), y(n), z(n))
          do i=1, n 
         read(10,*)x(i), y(i), z(i)
        enddo
         close(10)

          open(1, file=outfile, form='unformatted', access="stream")
         write(1) n
write(1) (x(i),i=1,n)
write(1) (y(i),i=1,n)
write(1) (z(i),i=1,n)
         close(1)



           end
