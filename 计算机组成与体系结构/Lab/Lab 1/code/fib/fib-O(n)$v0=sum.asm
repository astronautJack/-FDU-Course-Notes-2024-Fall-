#File: fib-O(n).asm
    .text
main:
    li $v0 5          #Input n
	syscall
	move $a0 $v0
	jal fib           #Call fib
	move $s0 $v0      #Output nth Fibonacci number
	li $v0 1          
	move $a0 $s0
	syscall
	li $v0 10         #Terminate main
	syscall
fib:
    addi $v0 $0 1
    addi $t0 $0 3     #If n<3, return 1
	slt $t1 $a0 $t0
	bne $t1 $0 done
	addi $t2 $0 1     #int prev=curr=1,
	addi $t3 $0 1
	addi $t0 $0 2     #int i=2
for:                  #for(int i=2;i<n;i++)
    slt $t1 $t0 $a0
    beq $t1 $0 done
	add $v0 $t2 $t3   #$v0=sum=prev+curr=$t2+$t3
	move $t2 $t3      #$t2=prev=curr=$t3
	move $t3 $v0      #$t3=curr=sum=$v0
	addi $t0 $t0 1
	j for
done:
	jr $ra
    
