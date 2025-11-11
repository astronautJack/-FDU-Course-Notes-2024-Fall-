`timescale 1ns / 1ps
// Create Date: 2024/04/26 10:53:34
// Module Name: InstructionRegister
// Description: Ö¸Áî¼Ä´æÆ÷IR
module InstructionRegister (input logic        clk, Reset,
                            input logic        IRWriteEn,
                            input logic [31:0] NextInstruction,
                            output logic [31:0] CurrentInstruction);
    
    always_ff @(posedge clk or posedge Reset)
    begin
        if (Reset) 
            CurrentInstruction <= 32'h00000000;
        else if (IRWriteEn) 
            CurrentInstruction <= NextInstruction;
    end
    
endmodule
