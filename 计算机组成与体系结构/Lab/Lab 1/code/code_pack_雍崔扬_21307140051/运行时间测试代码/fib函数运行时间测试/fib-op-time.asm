#File: fib-tail_recur.asm
#Register used:
#   $a0 -- n
#   $a1 -- first
#   $a2 -- second
#Main idea:
#   Always using the last two fibonacci number at present to recur
#   n can be regarded as the remaining recursion degree, i.e. there is still n times to go
    .data
	    msg1:    .asciiz "Finish!"
		msg2:    .asciiz "\n"
    .text
main:
    li $v0 5                # Input n
	syscall
	move $s0 $v0
	li $s1 200000
	move $s2 $0
loop:
    slt $s3 $s2 $s1         # for(i=0;i<200000;i++)
	beq $s3 $0 exit
    move $a0 $s0
	addi $a1 $0 1           # first=1, second=1
	addi $a2 $0 1
	jal fib                 # Call fib(n, first, second)
	addi $s2 $s2 1
	j loop
exit:
    move $a0 $v0            # Print nth fibonacci number
    li $v0 1
	syscall
	li $v0 4                # Change line
	la $a0 msg2
	syscall
	li $v0 4                # Cry out "Finish!"
	la $a0 msg1
	syscall
	li $v0 10               # Terminate main
	syscall

fib:
	bne $a0 $0 fib_recurse  # If(n==0), return(first)

	move $v0 $a1
	jr $ra
	                        # Otherwise, return(fib(n-1, second, first+second))
fib_recurse:
    addi $sp $sp -4         # Save $ra by creating a 4-byte stack
	sw $ra 0($sp)

	addi $a0 $a0 -1         # n--
	add $a2 $a2 $a1         # second2 = first1 + second1
	sub $a1 $a2 $a1         # first2 = second2 - first1 = second1
	jal fib                 # Call fib(n-1, second, first+second)

	lw $ra 0($sp)           # Restore $ra
	addi $sp $sp 4
    jr $ra                  # return(fib(n-1, second, first+second))
