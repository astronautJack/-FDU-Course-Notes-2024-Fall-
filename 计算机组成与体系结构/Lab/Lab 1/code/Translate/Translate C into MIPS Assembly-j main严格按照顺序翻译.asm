#File:  Translate C into MIPS Assembly.asm
#Register used:
#    in main:
#$s0=initial address of arrs, $s1=N, $s2=result
#    in sumn:
#parameter: $a0=$s0, $a1=$s1
#local: $t0=idx, $t1=idx<N (i.e.$t0<$a1), $t2=address of arrs[idx], $t3=arrs[idx], $t4=sum
	.data
	    arrs:
		    .word    9, 7, 15, 19, 20, 30, 11, 18
			#int arrs[]={9, 7, 15, 19, 20, 30, 11, 18}                         
		msg:
		    .asciiz    "The result is: "
	.text
    j main
sumn:
	addi $t4 $0 0   #int sum=0
	addi $t0 $0 0   #int idx=0
for:
    slt $t1 $t0 $a1 #$t1=idx<N
	beq $t1 $0 done
	sll $t2 $t0 2   #$t2=idx*4+(int*)arrs=(int*)arrs[idx]
	add $t2 $t2 $a0 
	lw $t3 0($t2)   #$t3=arrs[idx]
	add $t4 $t4 $t3 #$t4=$t4+$t3, i.e.sum=sum+arrs[idx]
	addi $t0 $t0 1  #$t0=$t0+1, i.e.idx++
	j for
done:
    move $v0 $t4    #return $t4=sum
    jr $ra
main:
    la $s0 arrs     #$s0=initial address of arrs
	addi $s1 $0 8   #int N=8
	move $a0 $s0    #int result=sumn(arrs,n)
	move $a1 $s1
	jal sumn
	move $s2 $v0
	li $v0 4        #Print "The result is: "
	la $a0 msg
	syscall
	li $v0 1        #Print integer result
	move $a0 $s2
	syscall
    li $v0 10       #Terminate main
	syscall