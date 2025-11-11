`timescale 1ns / 1ps
// Create Date: 2024/04/30 14:03:42
// Module Name: RegisterFile
// Description: 寄存器文件
module RegisterFile(input logic        clk,
                    input logic        RegWriteEn,
                    input logic [4:0]  RegWriteAddr,
                    input logic [31:0] RegWriteData,
                    input logic [4:0]  RsAddr,
                    input logic [4:0]  RtAddr,
                    output logic [31:0] RsData,
                    output logic [31:0] RtData);
    
    logic [31:0] RegFile[31:0]; // 32个32位寄存器
    
    always_ff @(posedge clk)
    begin
        if (RegWriteEn) 
            RegFile[RegWriteAddr] <= RegWriteData; // 寄存器写使能为1时，写入数据
    end

    assign RsData = (RsAddr != 0) ? RegFile[RsAddr] : 32'h0; // 读出rs、rt中的数据
    assign RtData = (RtAddr != 0) ? RegFile[RtAddr] : 32'h0;

endmodule
