`timescale 1ns / 1ps
// Create Date: 2024/04/26 11:33:27
// Module Name: ProgramCounter
// Description: ³ÌÐò¼ÆÊýÆ÷
module ProgramCounter (input logic        clk, Reset, PCWriteEn,
                       input logic [31:0] NextPC,
                       output logic [31:0] PC);

    always_ff @(posedge clk or posedge Reset)
    begin
        if (Reset) 
            PC <= 32'h00400000;
        else if (PCWriteEn) 
            PC <= NextPC;
    end
    
endmodule
