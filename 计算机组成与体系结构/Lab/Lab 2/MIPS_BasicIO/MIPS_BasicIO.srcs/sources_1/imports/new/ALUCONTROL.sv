`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/04/10 21:29:13
// Design Name: 
// Module Name: ALUCONTROL
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


module ALUCONTROL(
    input  logic [3:0] ALUctr,    // ALU control input specifying the operation.
    output logic       SUBctr,    // Subtraction control output.
    output logic       OFctr,     // Overflow control output.
    output logic [2:0] OPctr      // Operation control output specifying the specific ALU operation.
);

    logic [4:0] Controls;         // Combined control signals for ease of assignment.

    // The ALU control vector is decomposed into separate control signals for operation, subtraction, and overflow.
    assign {SUBctr, OFctr, OPctr} = Controls;

    // Decide the control signal outputs based on the ALU control input.
    always_comb begin // Using always_comb for combinational logic to clearly indicate intent.
        case (ALUctr)
            4'b0000: Controls = 5'b01000; // add, addi
            4'b1000: Controls = 5'b11000; // sub
            4'b0001: Controls = 5'b00001; // and, andi
            4'b0010: Controls = 5'b00010; // or, ori
            4'b0011: Controls = 5'b00011; // nor, nori
            4'b0101: Controls = 5'b10101; // slt, slti
            4'b0100: Controls = 5'b00100; // sll (shift left logical)
            4'b0110: Controls = 5'b00110; // srl (shift right logical)
            4'b0111: Controls = 5'b00111; // sra (shift right arithmetic)
            4'b1110: Controls = 5'b10000; // beq, bne 
            4'b1111: Controls = 5'b00000; // lw, sw 
            default: Controls = 5'b00000; // Default case for unexpected input
        endcase
    end

endmodule
