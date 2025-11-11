`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/04/15 12:07:38
// Design Name: 
// Module Name: Top
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


module Top(
    input         logic clk, Reset,        // 输入：时钟信号和复位信号
    output logic [31:0] DMWriteData,       // 输出：数据内存写数据
    output logic [31:0] DMDataAddr,        // 输出：数据内存地址
    output        logic DMWriteEn          // 输出：数据内存写使能信号
);
    logic [31:0] PC, Instruction, DMReadData;
    
    // instantiate processor and memories
    MIPS mips(clk, Reset, Instruction, DMReadData, PC, DMWriteEn, DMDataAddr, DMWriteData);
    
    // DMDataAddr = ALUresult in MIPS module
    InstructionMemory IM(PC[7:2], Instruction);
    DataMemory DM(clk, DMWriteEn, DMDataAddr, DMWriteData, DMReadData);
    
endmodule
