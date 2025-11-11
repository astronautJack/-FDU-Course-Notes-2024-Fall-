`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/04/10 21:26:23
// Design Name: 
// Module Name: MUX6X32
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


module MUX6X32(
    input  logic [31:0] Y_addsub, Y_and, Y_or, Y_nor, Y_slt, Y_shift,   
	// 输入通道: 6个32位输入
    input  logic [2:0]  OPctr,                                             
	// 控制信号: 选择哪个输入通道的输出
    output logic [31:0] Y                                               
	// 输出通道: 选定的输入通道的数据输出
);

    // 使用always_comb代替always @*，增强代码的可读性和意图表达。
    // always_comb是SystemVerilog中引入的，表示逻辑应该在输入变化时立即重新计算。
    always_comb begin
        case (OPctr)
            3'b000: Y = Y_addsub; // add, addi, sub, lw, sw, beq, bne操作选择该通道
            3'b001: Y = Y_and;    // and, andi操作选择该通道
            3'b010: Y = Y_or;     // or, ori操作选择该通道
            3'b011: Y = Y_nor;    // nor, nori操作选择该通道
            3'b101: Y = Y_slt;    // slt, slti操作选择该通道
            3'b100: Y = Y_shift;  // 逻辑左移sll操作选择该通道
            3'b110: Y = Y_shift;  // 逻辑右移srl操作选择该通道
            3'b111: Y = Y_shift;  // 算术右移sra操作选择该通道
            default: Y = 32'bx;   // 对未定义的控制信号，输出不定值
        endcase
    end

endmodule
