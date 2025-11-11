#File:  Translate C into MIPS Assembly.asm
    .data
	    arrs:
		    .space    32      #open up 32 bytes of space to store
			                  #the array "arrs" which contains 8 integers
		msg:
		    .asciiz "The result is:"
	.text

main:
    #$s1=N, $s2=result
    la $s0 arrs     #将数组arrs的首地址存在$s0中
	addi $t0 $0 9   #int arrs[]={9,7,15,19,20,30,11,18}
	addi $t1 $0 7
	addi $t2 $0 15
	addi $t3 $0 19
	addi $t4 $0 20
	addi $t5 $0 30
	addi $t6 $0 11
	addi $t7 $0 18
	sw $t0 0($s0)
	sw $t1 4($s0)
	sw $t2 8($s0)
	sw $t3 12($s0)
	sw $t4 16($s0)
	sw $t5 20($s0)
	sw $t6 24($s0)
	sw $t7 28($s0)
	addi $s1 $0 8   # int N=8
	addi $a0 $s0 0  # int result=sumn(arrs,n)
	addi $a1 $s1 0
	jal sumn
	add $s2 $v0 $0
	li $v0 4        #output the message "The result is:"
	la $a0 msg
	syscall
	li $v0 1        #output the integer "result"
	move $a0 $s2
	syscall
    li $v0 10       #terminate main
	syscall
sumn:
    #$a0=int* arrs, $a1=N
    #$v0=sum, $t0=idx, $t1=idx<N (i.e.$t0<$a1), $t2=(int*)arrs[idx], $t3=arrs[idx]
	addi $v0 $0 0   #int sum=0
	addi $t0 $0 0   #int idx=0
for:
    slt $t1 $t0 $a1 #$t1=idx<N
	beq $t1 $0 done
	sll $t2 $t0 2   #$t2=idx*4
	add $t2 $t2 $a0 #$t2=$t2+$a0=idx*4+(int*)arrs=arrs[idx]
	lw $t3 0($t2)   #$t3=arrs[idx]
	add $v0 $v0 $t3 #$v0=$v0+$t3, i.e.sum=sum+arrs[idx]
	addi $t0 $t0 1  #$t0=$t0+1, i.e.idx++
	j for
done:
    jr $ra          #return sum=$v0