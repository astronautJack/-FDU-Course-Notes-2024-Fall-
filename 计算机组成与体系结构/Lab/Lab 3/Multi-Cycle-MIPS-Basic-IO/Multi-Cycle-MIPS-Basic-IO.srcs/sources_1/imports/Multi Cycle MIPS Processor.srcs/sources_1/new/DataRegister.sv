`timescale 1ns / 1ps
// Create Date: 2024/04/30 10:53:34
// Module Name: DataRegister
// Description: Êý¾Ý¼Ä´æÆ÷DR\A\B
module DataRegister (input logic        clk, Reset,
                     input logic [31:0] NextData,
                     output logic [31:0] CurrentData);
    
    always_ff @(posedge clk or posedge Reset)
    begin
        if (Reset) 
            CurrentData <= 32'h0;
        else       
            CurrentData <= NextData;
    end
    
endmodule
