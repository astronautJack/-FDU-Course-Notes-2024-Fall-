`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/04/10 21:34:25
// Design Name: 
// Module Name: EXTENDER
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

module EXTENDER(input  [15:0] Imm16,
                input         ExtOp,
                output [31:0] Imm32);
    logic [31:0] A0,A1;
    assign A0 = {16'b0,Imm16};//0拓展
    assign A1 = {{16{Imm16[15]}},Imm16};//符号拓展
    MUX2X32 MuxExtentionType(A0,A1,ExtOp,Imm32);//根据ExtOp选择0拓展还是符号拓展
endmodule
