#File: fib-O(n).asm
    .text
main:
    li $v0 5          #Input n
	syscall
	move $a1 $v0

	jal fib           #Call fib
	move $s0 $v0

	li $v0 1          #Output nth Fibonacci number
	move $a0 $s0
	syscall

	li $v0 10         #Terminate main
	syscall

fib:
    addi $t3 $0 1
    addi $t4 $0 2     #If n<2, return 1
	slt $t0 $a1 $t4
	bne $t0 $0 done
	addi $t1 $0 1     #int first=second=1, i=1
	addi $t2 $0 1
	addi $t0 $0 1
for:                  #for(int i=1; i<n; i++)
    slt $t4 $t0 $a1
    beq $t4 $0 done

	add $t4 $t2 $t3   #$t3=third=first+second=$t1+$t2
	move $t2 $t3      #$t1=first=second=$t2
	move $t3 $t4      #$t2=second=third=$t3

	addi $t0 $t0 1
	j for
done:
    move $v0 $t3
	jr $ra
    
