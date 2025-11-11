#File: fib-O(n).asm
    .data
	    msg0: .asciiz "n="
		msg1: .asciiz "\t"
	    msg2: .asciiz "\n"
    .text
main:
loop:
    li $v0 5          #Input n
	syscall
	move $a1 $v0
	jal fib           #Call fib
	li $v0 10         #Terminate main
	syscall
fib:
    addi $t0 $0 3     #If n<3, return 1
	slt $t1 $a1 $t0
	bne $t1 $0 done
	addi $t2 $0 1     #int prev=curr=1,
	addi $t3 $0 1
	addi $t0 $0 2     #int i=2
for:                  #for(int i=2;i<n;i++)
    slt $t1 $t0 $a1
    beq $t1 $0 done
	add $t4 $t2 $t3   #$t4=sum=prev+curr=$t2+$t3
	move $t2 $t3      #$t2=prev=curr=$t3
	move $t3 $t4      #$t3=curr=sum=$t4
	addi $t0 $t0 1
	
	#printf("n=%d\t%d\n",i+1,sum)
	li $v0 4
	la $a0 msg0
	syscall
	li $v0 1
	move $a0 $t0
	syscall
	li $v0 4
	la $a0 msg1
	syscall
	li $v0 1
	move $a0 $t4
	syscall
	li $v0 4
	la $a0 msg2
	syscall

	j for
done:
	jr $ra
    
