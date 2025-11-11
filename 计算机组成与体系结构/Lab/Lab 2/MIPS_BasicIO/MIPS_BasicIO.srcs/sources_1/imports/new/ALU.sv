`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/04/10 21:31:40
// Design Name: 
// Module Name: ALU
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module ALU(input  logic [31:0] A, B,
           input  logic [3:0]  ALUctr,
           output logic [31:0] Y,      //ALUresult
           output logic 	   OF, ZF);

    logic [31:0] Y_addsub, Y_and, Y_or, Y_nor, Y_slt, Y_shift;
    logic 		 SUBctr, OFctr;
    logic [2:0]  OPctr;
    
	// ALU操作控制器
    ALUCONTROL ALUcontrol(ALUctr, SUBctr, OFctr, OPctr);
    
    // 32位并行进位加法器
    CLA32 getY_addsub(A, B, SUBctr, Y_addsub, Add_Overflow, Add_Sign, ZF, Add_Carry);
    assign Y_and = A & B;
    assign Y_or = A | B;
    assign Y_nor = ~(Y_or);
    assign Y_slt = (Add_Overflow ^ Add_Sign) ? 1 : 0;//执行带符号整数比较小于0
    
    // 32位移位器
    Shifter getY_shift(B, A[10:6], OPctr[1], OPctr[0], Y_shift);
    
    // 32位6路选择器
    MUX6X32 mux6x32(Y_addsub, Y_and, Y_or, Y_nor, Y_slt, Y_shift, OPctr, Y);
    
    assign OF = OFctr & Add_Overflow;
    
endmodule
