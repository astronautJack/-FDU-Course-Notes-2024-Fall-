# File:         p1.asm
# Written by:   Larry Merkle, Dec. 4, 2002
# Modified by:  J.P. Mellor, 3 Sep. 2004
# Modified by:  J.P. Mellor, 2 Dec. 2008
#
# This file contains a MIPS assembly language program that uses only the 
# following instructions:
#
#   ori rt, rs, imm	- Puts the bitwise OR of register rs and the
#			  zero-extended immediate into register rt
#   add rd, rs, rt	- Puts the sum of registers rs and rt into register rd.
#   syscall		- Register $v0 contains the number of the system
#			  call provided by SPIM (when $v0 contains 10,
#			  this is an exit statement).
#
# It calculates 40 + 17.
#
# It is intended to help CSSE 232 students familiarize themselves with MIPS
# and SPIM.
				
	.globl main		# Make main globl so you can refer to
				# it by name in SPIM.

	.text			# Text section of the program
				# (as opposed to data).

main:				# Program starts at main.
la $t0, value
li $v0, 5
syscall
sw $v0, 0($t0)
li $v0, 5
syscall
sw $v0, 4($t0)
lw $t1, 0($t0)
lw $t2, 4($t0)
add $t3, $t1, $t2
sw $t3, 8($t0)
li $v0, 4
la $a0, msg1
syscall
li $v0, 1
move $a0, $t3
syscall
li $v0 10
syscall

.data

msg1: .ascii "result="
value: .word 0