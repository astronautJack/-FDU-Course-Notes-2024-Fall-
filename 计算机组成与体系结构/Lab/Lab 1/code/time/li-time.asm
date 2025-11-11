    .data
	    msg:  .asciiz "Finish!"
	.text
main:
    li $s0 0x0000FFFF
	li $t0 50000000
	addi $t1 $0 0
loop:
    slt $t2 $t1 $t0          #for(i=0;i<50000000;i++) 
	beq $t2 $0 exit

	li $s1 0x0000FFFF             #Core code
	li $s1 0

	addi $t1 $t1 1
	j loop
exit:
    li $v0 4
	la $a0 msg
	syscall
    li $v0 10
	syscall