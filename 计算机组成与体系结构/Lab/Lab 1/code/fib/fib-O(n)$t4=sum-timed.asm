#File: fib-O(n).asm
    .text
main:
    li $v0 5          #Input n
	syscall
	move $a0 $v0
	li $v0 30         #Starting time
	syscall
	jal fib           #Call fib
	li $v0 30         #Fininshing time
	syscall
	sub $s0 $a1 $a0   #Subtract time values
	li $v0 1          #Print time taken
	move $a0 $t2
	syscall
	li $v0 10         #Terminate main
	syscall
fib:
    addi $t4 $0 1
    addi $t0 $0 3     #If n<3, return 1
	slt $t1 $a0 $t0
	bne $t1 $0 done
	addi $t2 $0 1     #int prev=curr=1,
	addi $t3 $0 1
	addi $t0 $0 2     #int i=2
for:                  #for(int i=2;i<n;i++)
    slt $t1 $t0 $a0
    beq $t1 $0 done
	add $t4 $t2 $t3   #$t4=sum=prev+curr=$t2+$t3
	move $t2 $t3      #$t2=prev=curr=$t3
	move $t3 $t4      #$t3=curr=sum=$t4
	addi $t0 $t0 1
	j for
done:
	li $v0 1          #Output nth Fibonacci number
	move $a0 $t4
	syscall
	jr $ra
    
