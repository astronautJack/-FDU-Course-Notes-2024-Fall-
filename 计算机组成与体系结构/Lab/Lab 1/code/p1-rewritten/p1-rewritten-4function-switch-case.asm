# File:p1-rewritten.asm
# After receiving two integers,calculate the sum and output it. Loop if next input is 0.
    .data
msg1: .asciiz "Please enter 1st number:"
msg2: .asciiz "Please enter 2nd number:"
msg3: .asciiz "The result of "
msg4: .asciiz " & "
msg5: .asciiz " is:"
msg6: .asciiz "\n"
msg7: .asciiz "Do you want to try another(0_continue/1_exit):"
msg8: .asciiz "EXIT!"
	.text
main:
loop:
    la $a0 msg1      #Print "Please enter 1st number:"
    jal print_get
	move $t1 $v0     #Get 1st number into $t1

    la $a0 msg2      #Print "Please enter 2nd number:"
    jal print_get
	move $t2 $v0     #Get 2nd number into $t2

	add $t3 $t1 $t2  #Calculate $t3=$t1+$t2

    la $a0 msg3      #Print "The result of " and 1st number
	move $a1 $t1
	jal print_print
    
	la $a0 msg4      #Print " & " and 2nd number
	move $a1 $t2
	jal print_print
	
    la $a0 msg5      #Print " is:" and sum
    move $a1 $t3
	jal print_print

	jal change_line  #Change_line
    
	jal continue_or_exit  #Judge whether to continue or exit

EXIT:
	li $v0 10
	syscall

change_line:         #Change line by outputing "\n"
    li $v0 4
	la $a0 msg6
	syscall
	jr $ra

print_get:           #Print a string and get an integer
    li $v0 4
	syscall
	li $v0 5
	syscall
	jr $ra

print_print:         #Print a string and print an integer
    li $v0 4
	syscall
	move $a0 $a1
	li $v0 1
	syscall
	jr $ra

continue_or_exit:    #Judge whether to continue or exit
	keep_on_asking:
    la $a0 msg7      #Print "Do you want to try another(0_continue/1_exit):"
	jal print_get
	move $t4 $v0     #Get user's instruction
case0:
	bne $t4 $0 case1 #if input==0 change line and loop
    jal change_line
	j loop
case1:               #else if input==1, print "EXIT!" and exit
	addi $t5 $0 1
	bne $t4 $t5 default
	li $v0 4                        
	la $a0 msg8
	syscall
	j EXIT
default:             #else keep on asking 
    j keep_on_asking
