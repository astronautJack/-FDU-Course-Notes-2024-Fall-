`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/04/10 22:09:10
// Design Name: 
// Module Name: DataPath
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


module DataPath(
    input  logic 	    clk, Reset,
    input  logic [31:0] Instruction,
    input  logic [31:0] DMReadData,
    input  logic [1:0]  Branch, // 是否分支,以及是beq还是bne
    input  logic 		Jump, // 是否跳转
    input  logic 		RegDst, // 目标寄存器是rt还是rd
    input  logic 		ALUsrc, // 控制B的来源是Rt还是Imm16
    input  logic 		MemToReg, // 选择将ALU结果还是存储器的输出写入寄存器
    input  logic 		RegWriteEn, //寄存器写使能信号，只有写使能为1且不溢出的情况下才写入
    input  logic 		DMWriteEn, //存储器写使能信号，用的是addu，不判断溢出
    input  logic 		ExtOp, // 决定Imm16做0拓展还是符号拓展
    input  logic [3:0]  ALUctr, // ALU控制码
    output logic [31:0] PC,
    output logic [31:0] ALUresult, DMWriteData
);
	// Next PC Logic
    logic [15:0] Imm16;
    assign Imm16 = Instruction[15:0];
    
    logic [25:0] Target26;
    assign Target26 = Instruction[25:0];
    
    logic [31:0] NewPC;
    PC PCregister(clk, Reset, NewPC, PC);
    
    logic ZF;
	PCadder pcadder(PC, Imm16, Target26, Jump, ZF, Branch, NewPC);

    // Register File Logic
    logic [4:0] RsAddr, RtAddr, RdAddr, RegWriteAddr;
    assign {RsAddr, RtAddr, RdAddr} = Instruction[25:11];
    
    logic [31:0] InputA, busB, InputB;
    logic [31:0] RegWriteData;
    logic OF;
    RegisterFile RF(clk, RegWriteEn & (!OF), RegWriteAddr, RegWriteData, RsAddr, RtAddr, InputA, busB); // InputA = RsData, busB = RtData;
   	
    assign DMWriteData = busB;
    MUX2X5 DestinationRegister(RtAddr, RdAddr, RegDst, RegWriteAddr);// 根据RegDst选择目的寄存器是Rt还是Rd
    MUX2X32 RegisterWriteData(ALUresult, DMReadData, MemToReg, RegWriteData); // 根据MemToReg选择写入寄存器的数据是ALUresult还是DMReadData

    // ALU Logic
    logic [31:0] Imm32;
    EXTENDER Imm16Extend(Imm16, ExtOp, Imm32);// 根据ExtOp选择对Imm16做0拓展还是符号拓展
    MUX2X32 MuxInputB(busB, Imm32, ALUsrc, InputB);// 根据ALUsrc选择InputB是busB还是Imm32
    ALU ArithLogicUnit(InputA, InputB, ALUctr, ALUresult, OF, ZF);

endmodule