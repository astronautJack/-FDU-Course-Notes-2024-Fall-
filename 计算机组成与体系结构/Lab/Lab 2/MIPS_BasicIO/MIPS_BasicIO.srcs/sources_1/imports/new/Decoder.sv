`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/04/10 23:31:47
// Design Name: 
// Module Name: Decoder
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


module Decoder(
    input  logic [31:0] Instruction, // 32位指令
    output logic [1:0]  Branch, // 是否分支，以及是beq还是bne
    output logic        Jump, // 是否跳转
    output logic        RegDst, // 目标寄存器是rt还是rd
    output logic        ALUsrc, // 控制B的来源是Rt还是Imm16
    output logic        MemToReg, // 选择将ALU结果还是存储器的输出写入寄存器
    output logic        RegWriteEn, // 寄存器写使能信号，只有写使能为1且不溢出的情况下才写入
    output logic        DMWriteEn, // 存储器写使能信号，用的是addu，不判断溢出
    output logic        ExtOp, // 决定Imm16做0拓展还是符号拓展
    output logic [3:0]  ALUctr // ALU控制码
);

    logic [5:0] op, func;
    assign {op, func} = {Instruction[31:26], Instruction[5:0]};
    logic [3:0] ALUctr0;
    logic [13:0] Controls;
    logic JudgeRType, RegWriteEn0;

    assign {Branch, Jump, RegDst, ALUsrc, MemToReg, RegWriteEn0, DMWriteEn, ExtOp, ALUctr0, JudgeRType} = Controls;

    always_comb begin
        case(op)
            6'b000000: Controls = 14'b00_0100100_0000_1; // R-Type
            6'b001000: Controls = 14'b00_0010101_0000_0; // addi
            6'b001100: Controls = 14'b00_0010100_0001_0; // andi
            6'b001101: Controls = 14'b00_0010100_0010_0; // ori
            6'b001010: Controls = 14'b00_0010101_0101_0; // slti
            6'b101011: Controls = 14'b00_0010011_1111_0; // sw
            6'b100011: Controls = 14'b00_0011101_1111_0; // lw
            6'b000100: Controls = 14'b10_0000000_1110_0; // beq
            6'b000101: Controls = 14'b11_0000000_1110_0; // bne
            6'b000010: Controls = 14'b00_1000000_0000_0; // j
            default:   Controls = 14'b00_0000000_0000_0; // others
        endcase
    end

    // 使用SystemVerilog的内联函数定义
    function automatic [4:0] alter(input logic [5:0] func);
        case(func)
            6'b100000: alter = 5'b1_0000; // add 0000
            6'b100010: alter = 5'b1_1000; // sub 1000
            6'b100100: alter = 5'b1_0001; // and 0001
            6'b100101: alter = 5'b1_0010; // or 0010
            6'b100111: alter = 5'b1_0011; // nor 0011
            6'b101010: alter = 5'b1_0101; // slt 0101
            6'b000000: alter = 5'b1_0100; // sll 0100, 包括nop, 因为 nop = sll $0 $0 0
            6'b000010: alter = 5'b1_0110; // srl 0110
            6'b000011: alter = 5'b1_0111; // sra 0111
            default:   alter = 5'b0_0000; // 其他情况置寄存器写使能为0
        endcase
    endfunction

    logic [4:0] Alter, Final;
    assign Alter = alter(func);
    MUX2X5 AlterRType({RegWriteEn0, ALUctr0}, Alter, JudgeRType, Final);
    assign {RegWriteEn, ALUctr} = Final;

endmodule
