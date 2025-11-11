`timescale 1ns / 1ps
// Create Date: 2024/04/26 11:12:35
// Module Name: Extender
// Description: 扩展器
module Extender(input logic [15:0] Imm16,
                input logic        ExtOp,
                output logic [31:0] Imm32);
    
    logic [31:0] A0, A1;

    assign A0 = {16'b0, Imm16};        // 0扩展
    assign A1 = {{16{Imm16[15]}}, Imm16}; // 符号扩展
    
    MUX2X32 MuxExtentionType(.A0(A0), .A1(A1), .S(ExtOp), .Y(Imm32));
    // 根据ExtOp选择0扩展还是符号扩展

endmodule
