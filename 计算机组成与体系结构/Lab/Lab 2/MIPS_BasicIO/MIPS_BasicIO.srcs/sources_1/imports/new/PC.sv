`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/04/10 21:56:00
// Design Name: 
// Module Name: PC
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


module PC (
    input  logic 		clk, Reset,
    input  logic [31:0] NewPC,
    output logic [31:0] PC
);

    always_ff @(posedge clk or posedge Reset)
        PC <= (Reset) ? 0 : NewPC;

endmodule
