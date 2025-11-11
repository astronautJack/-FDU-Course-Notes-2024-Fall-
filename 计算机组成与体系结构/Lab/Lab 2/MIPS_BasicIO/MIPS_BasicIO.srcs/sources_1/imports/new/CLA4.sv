`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/04/10 20:48:20
// Design Name: 
// Module Name: CLA4
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


module CLA4(input  logic [3:0] A, B,
    		input         	   Cin,
    		output        	   Cout,
    		output logic [3:0] Y);

    /* Intermediate signals:
	   		Propagate Signal: p
	   		Generate Signal	: g  */
    logic [3:0] p, g;

    // Calculate p and g signals
    assign p = A ^ B;  // XOR
    assign g = A & B;

    // Array P and G
    logic [3:0] P, G;

    /* Calculate P and G arrays
       P[i] is the product of p[0] to p[i], G[i] = g[i] + p[i] * G[i-1] */
    assign {P[0], G[0]} = {p[0], g[0]};
    assign {P[1], G[1]} = {p[1] & P[0], g[1] | (p[1] & G[0])};
    assign {P[2], G[2]} = {p[2] & P[1], g[2] | (p[2] & G[1])};
    assign {P[3], G[3]} = {p[3] & P[2], g[3] | (p[3] & G[2])};

    // Array for carry signals
    logic [4:0] C;

    /* Calculate carry signals
       C[0] is the initial carry-in
       C[4:1] is calculated using the formulas from the textbook */
    assign C[0] = Cin;
    assign C[4:1] = G | (P & {4{Cin}});

    // Calculate the carry-out
    assign Cout = C[4];

    // Calculate the sum output
    assign Y = p ^ C[3:0];

endmodule