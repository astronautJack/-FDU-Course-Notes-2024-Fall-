`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/04/10 21:45:58
// Design Name: 
// Module Name: RegisterFile
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


module RegisterFile(
    input         clk,
    input         RegWriteEn,
    input  [4:0]  RegWriteAddr,
    input  [31:0] RegWriteData,
    input  [4:0]  RsAddr,
    input  [4:0]  RtAddr,
    output [31:0] RsData,
    output [31:0] RtData
);

    logic [31:0] RegFile[31:0]; // 32个32位寄存器

    // 寄存器写使能为1时，写入数据
    always_ff @(posedge clk) begin
        if (RegWriteEn)
            RegFile[RegWriteAddr] <= RegWriteData; 
    end

    // 使用三目运算符进行条件赋值，如果地址为0，则输出0；否则输出寄存器值
    assign RsData = (RsAddr != 0) ? RegFile[RsAddr] : 0; // 读出rs中的数据
    assign RtData = (RtAddr != 0) ? RegFile[RtAddr] : 0; // 读出rt中的数据
endmodule
