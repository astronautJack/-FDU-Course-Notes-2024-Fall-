`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/04/10 21:52:29
// Design Name: 
// Module Name: PCadder
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


module PCadder(
    input  logic [31:0] PC,
    input  logic [15:0] Imm16,
    input  logic [25:0] Target26,
    input  logic 		Jump,
    input  logic 		ZF,
    input  logic [1:0]  Branch,
    output logic [31:0] NewPC
);
    // 跳转目标地址
    logic [31:0] JumpPC;
    assign JumpPC = {PC[31:28], Target26, 2'b0};

    // 顺序执行地址
    logic [31:0] PCadd4;
    CLA32 adder1(PC, 32'b100, 1'b0, PCadd4, Null1, Null2, Null3, Null4);

    // Imm16符号扩展为最后两位为0的32位数
    logic [31:0] Imm16Extend;
    assign Imm16Extend = {{14{Imm16[15]}}, Imm16, 2'b0};

    // 计算分支目标地址
    logic [31:0] BranchPC;
    CLA32 adder2(PCadd4, Imm16Extend, 1'b0, BranchPC, Null1, Null2, Null3, Null4);

    // 判断是否分支
    logic WhetherToBranch;
    assign WhetherToBranch = Branch[1] & (Branch[0] ^ ZF);

    logic [31:0] MiddlePC;
    // 根据WhetherToBranch选择PC+4还是BranchPC
    MUX2X32 MuxBranch(PCadd4, BranchPC, WhetherToBranch, MiddlePC);

    // 根据Jump选择MiddlePC还是JumpPC作为新的PC值
    MUX2X32 MuxJump(MiddlePC, JumpPC, Jump, NewPC);
endmodule
