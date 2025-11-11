`timescale 1ns / 1ps
// Create Date: 2024/05/06 14:39:09
// Module Name: MIPS
// Description: 多周期MIPS处理器
module MIPS(input  logic        clk,Reset,
            input  logic [31:0] MemReadData,
            output logic        MemWriteEn,IorD,
            output logic [31:0] MemAddr,MemWriteData,PC);
    
    logic       ExtOp,PCWriteEn,IRWriteEn,RegWriteEn,MemToReg,RegDst;// IorD 和 MemWriteEn 已经声明了
    logic [1:0] ALUsrcA,ALUsrcB,PCsrc,PCWrCond;
    logic [3:0] ALUctr;
    logic [5:0] op,funct;
    
    Decoder DEC(clk,Reset,op,funct,//input
                ExtOp,PCWriteEn,MemWriteEn,IRWriteEn,//output
                RegWriteEn,IorD,MemToReg,RegDst,
                ALUsrcA,ALUsrcB,
                PCsrc,PCWrCond,
                ALUctr);
                
    DataPath DP(clk,Reset,MemReadData,//input
                ExtOp,PCWriteEn,IRWriteEn,
                RegWriteEn,IorD,MemToReg,RegDst,
                ALUsrcA,ALUsrcB,
                PCsrc,PCWrCond,
                ALUctr,
                op,funct,MemWriteData,MemAddr,PC);//output

endmodule
