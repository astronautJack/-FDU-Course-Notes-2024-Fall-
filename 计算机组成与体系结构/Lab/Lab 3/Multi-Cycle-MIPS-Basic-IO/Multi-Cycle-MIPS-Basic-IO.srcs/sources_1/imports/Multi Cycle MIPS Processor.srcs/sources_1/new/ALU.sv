`timescale 1ns / 1ps
// Create Date: 2024/04/12 11:07:52
// Module Name: ALU
// Description: ALU运算器
module ALU(input logic [31:0] A,B,
           input logic [3:0]  ALUctr,
           output logic [31:0] ALUresult,
           output logic        OF,ZF);
    logic [31:0] Y_addsub,Y_and,Y_or,Y_nor,Y_slt,Y_shift;
    logic SUBctr,OFctr,SIGctr,SLTctr;
    logic [2:0] OPctr;
    ALUcontrol ALUcontrol(ALUctr,SUBctr,OFctr,SIGctr,OPctr);//ALU操作控制器
    CLA32 getY_addsub(A,B,SUBctr,Y_addsub,Add_Overflow,Add_Sign,ZF,Add_Carry);
    assign Y_and = A & B;
    assign Y_or = A | B;
    assign Y_nor =  ~(Y_or);
    MUX2X1 SetLessThan(SUBctr ^ Add_Carry,Add_Overflow ^ Add_Sign,SIGctr,SLTctr);
    //SIGctr控制slt执行带符号整数比较小于0还是无符号整数比较小于0
    assign Y_slt = (SLTctr) ? 1 : 0;
    Shifter getY_shift(B,A[10:6],OPctr[1],OPctr[0],Y_shift);
    MUX6X32 getALUresult(Y_addsub,Y_and,Y_or,Y_nor,Y_slt,Y_shift,OPctr,ALUresult);//32位6路选择器
    assign OF = OFctr & Add_Overflow;
endmodule