`timescale 1ns / 1ps
// Create Date: 2023/05/05 19:28:17
// Module Name: DataPath
// Description: 数据通路
module DataPath(input  logic        clk, Reset,
                input  logic [31:0] MemReadData,//从存储器读出的数据
                input  logic        ExtOp,PCWriteEn,IRWriteEn,
                input  logic        RegWriteEn,IorD,MemToReg,RegDst,
                input  logic [1:0]  ALUsrcA,ALUsrcB,PCsrc,PCWrCond,
                input  logic [3:0]  ALUctr,
                //Decoder产生的控制码里，只有MemWriteEn不在DataPath中
                output logic [5:0]  op,funct,
                output logic [31:0] MemWriteData,MemAddr,PC);
                //写入存储器的数据、写入|读取存储器的地址
    //Preparation
    logic [31:0] Instruction,MemReadData_0;
    //MemReadData_0是MDR（MemReadData Register）的输出端
    InstructionRegister IR(clk,Reset,IRWriteEn,MemReadData,Instruction);
    DataRegister MDR(clk,Reset,MemReadData,MemReadData_0);
    assign {op,funct} = {Instruction[31:26],Instruction[5:0]};//output op\funct
    logic [15:0] Imm16;
    assign Imm16 = Instruction[15:0];
    logic [31:0] Imm32;
    Extender EXT(Imm16,ExtOp,Imm32);//因为设计的指令中有ori，故必须添加ExtOp控制码
    logic [25:0] Target26;
    assign Target26 = Instruction[25:0];
    logic [31:0] JumpPC;
    assign JumpPC = {PC[31:28],Target26,2'b00};//跳转目标地址
    logic [4:0] RsAddr,RtAddr,RdAddr;
    assign {RsAddr,RtAddr,RdAddr} = {Instruction[25:21],Instruction[20:16],Instruction[15:11]};
    //Next PC Logic
    logic [31:0] NextPC,ALUresult,PreviousALUresult;
    ProgramCounter PC_(clk,Reset,PCWriteEn,NextPC,PC);//output PC
    logic        Branch, ZF;
    assign Branch = PCWrCond[1] & (PCWrCond[0] ^ ZF);
    // PreviousALUresult 即为 BranchPC
    MUX4X32_PC NextPCsource(ALUresult,PreviousALUresult,JumpPC,PC,Branch,PCsrc,NextPC);
    //Register File Logic
    logic [31:0] RsData,RtData,DataA,DataB,SourceA,SourceB;
    logic        OF;
    logic [4:0]  RegWriteAddr;
    MUX2X5  get_RegWriteAddr(RtAddr,RdAddr,RegDst,RegWriteAddr);
    //目标寄存器地址，来源由RegDst控制
    logic [31:0] RegWriteData;
    MUX2X32 get_RegWriteData(PreviousALUresult,MemReadData_0,MemToReg,RegWriteData);
    //写入寄存器的数据，来源由MemToReg控制
    RegisterFile RF(clk,RegWriteEn & (!OF),RegWriteAddr,RegWriteData,RsAddr,RtAddr,RsData,RtData);
    DataRegister Data_A(clk,Reset,RsData,DataA);
    DataRegister Data_B(clk,Reset,RtData,DataB);
    assign MemWriteData = DataB;//output MemWriteData，写入存储器的数据
    //ALU Logic
    MUX3X32_Data get_SourceA(PC,DataA,Instruction,ALUsrcA,SourceA);
    MUX4X32_Data get_SourceB(DataB,4,Imm32,{Imm32[29:0],2'b0},ALUsrcB,SourceB);
    ALU ArithLogicUnit(SourceA,SourceB,ALUctr,ALUresult,OF,ZF);
    DataRegister get_PreviousALUresult(clk,Reset,ALUresult,PreviousALUresult);
    MUX2X32 get_MemAddr(PC,PreviousALUresult,IorD,MemAddr);
    //output MemAddr，写入\读取存储器的地址，来源由IorD控制
endmodule