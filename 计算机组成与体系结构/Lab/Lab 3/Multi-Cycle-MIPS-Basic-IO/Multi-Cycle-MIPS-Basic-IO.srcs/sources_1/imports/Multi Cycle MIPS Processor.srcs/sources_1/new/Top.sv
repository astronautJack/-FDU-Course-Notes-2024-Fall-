`timescale 1ns / 1ps
// Create Date: 2024/05/06 14:59:30
// Module Name: Top
// Description: CPU封装
module Top(input  logic        clk,Reset,
           output logic        MemWriteEn,IorD, // 输出的量用于仿真测试
           output logic [31:0] MemAddr,MemWriteData,PC,MemReadData);
    MIPS MIPS_(clk,Reset,MemReadData, // input
               MemWriteEn,IorD,MemAddr,MemWriteData,PC); // output
    Memory Memory_(clk,MemWriteEn,MemAddr,MemWriteData, // input
                   MemReadData); // output
endmodule
