# File:         p1-rewritten.asm
# After receiving two integers,calculate the sum and output it.
#
	.globl main		# Make main globl so you can refer to
                # it by name in SPIM.

	.text	    # Text section of the program
				# (as opposed to data).

main:				# Program starts at main.
    la $t0 value   #Save the initial address in $t0
loop:
	li $v0 4       #Output "Please enter 1st number:"
    la $a0 msg1
	syscall

    li $v0 5       #Input 1st number and copy it to $t1
	syscall
	move $t1 $v0

	li $v0 4        #Output "Please enter 2nd number:"
	la $a0 msg2
	syscall

	li $v0 5        #Input 2nd number and copy it to $t2
	syscall
	move $t2 $v0

	li $v0 4       #Output "The result of "
	la $a0 msg3
	syscall

	li $v0 1
	move $a0 $t1   #Output 1st number
	syscall

	li $v0 4        #Output " & "
	la $a0 msg4
	syscall

	li $v0 1         #Output 2nd number
	move $a0 $t2
	syscall

	li $v0 4        #Output " is:"
	la $a0 msg5
	syscall

	add $t3 $t1 $t2    #calculate $t1+$t2,and save the sum in $t3
	li $v0 1            #Output the sum
	move $a0 $t3
	syscall

	li $v0 4            #Change line by outputing "\n"
	la $a0 msg6
	syscall

	li $v0 4            #Output "Do you want to try another(0_continue/1_exit):"
	la $a0 msg7
	syscall

	li $v0 5            #Receive input 0 or 1 to decide whether to loop or exit
	syscall
    move $t4 $v0
	
	li $v0 4            #Change line by outputing "\n"
	la $a0 msg6
	syscall

	beq $t4 $0 loop    #if input==0 loop, else exit

	li $v0 10           #exit
	syscall

	.data

	value: .word 0
	msg1: .asciiz "Please enter 1st number:"
	msg2: .asciiz "Please enter 2nd number:"
	msg3: .asciiz "The result of "
	msg4: .asciiz " & "
	msg5: .asciiz " is:"
	msg6: .asciiz "\n"
	msg7: .asciiz "Do you want to try another(0_continue/1_exit):"
