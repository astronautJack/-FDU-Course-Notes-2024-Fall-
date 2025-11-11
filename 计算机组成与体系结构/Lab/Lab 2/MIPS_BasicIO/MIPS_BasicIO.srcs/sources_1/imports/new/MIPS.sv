`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/04/15 12:01:49
// Design Name: 
// Module Name: MIPS
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


module MIPS(
    input  logic 		clk, Reset,   // 输入：时钟信号clk和复位信号Reset
    input  logic [31:0] Instruction,  // 输入：从指令内存获取的32位指令
    input  logic [31:0] DMReadData,   // 输入：从数据内存读取的32位数据
    output logic [31:0] PC,           // 输出：程序计数器，指示当前执行的指令地址
    output logic 		DMWriteEn,    // 输出：数据内存写使能信号
    output logic [31:0] ALUresult,    // 输出：算术逻辑单元的计算结果
    output logic [31:0] DMWriteData   // 输出：写入数据内存的数据
);

    // Control signals: (DMWriteEn is already declared)
    logic [1:0] Branch;						// 分支类型
    logic Jump, RegDst, ALUsrc, MemToReg, RegWriteEn, ExtOp;
    /* 	Jump - 跳转信号
     	RegDst - 寄存器目标选择
     	ALUsrc - ALU操作数选择
     	MemToReg - 决定是否将数据内存的读取结果写回寄存器
     	RegWriteEn - 寄存器文件的写使能
     	ExtOp - 立即数扩展操作的控制信号 */
    logic [3:0] ALUctr;						// 定义ALU的操作类型
    
    // 解码器模块实例化：解析指令，输出控制信号到数据通路
    Decoder decoder(Instruction, Branch, Jump, RegDst, ALUsrc, MemToReg, RegWriteEn, DMWriteEn, ExtOp, ALUctr);
    
    // 数据通路模块实例化：执行指令的操作，包括ALU计算和数据存取
    DataPath DP(clk, Reset, Instruction, DMReadData, Branch, Jump, RegDst, ALUsrc, MemToReg, RegWriteEn, DMWriteEn, ExtOp, ALUctr, PC, ALUresult, DMWriteData);

endmodule 