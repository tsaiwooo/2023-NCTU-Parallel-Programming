CC = mpicxx
CFLAGS = -O3
all : matmul

matmul : main.cc function.cc
	${CC} ${CFLAGS} -o matmul main.cc function.cc

# %.o : %.cc
# 	${CC} $(CFLAGS) -c $<
clean :
	rm -rf matmul *.o 

send : 
	./send.sh

run : 
	mpirun -np 3 --hostfile hosts matmul < data1_1

