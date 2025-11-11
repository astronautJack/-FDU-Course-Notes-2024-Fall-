`timescale 1ns / 1ps
// Create Date: 2023/04/12 11:31:35
// Module Name: CLA4_Marked
// Description: 带溢出标志的4位并行进位加法器

module CLA4_Marked(
    input  logic [3:0] A, B,      // Input operands A and B
    input         	   Cin,       // Carry-in
    output        	   Cout,      // Carry-out
    output logic [3:0] Y,         // Sum output
    output        	   OF         // Overflow flag
);

    /* Intermediate signals: 
       Propagate Signal (p) and Generate Signal (g) */
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

    // Calculate the overflow flag
    assign OF = C[4] ^ C[3]; // Overflow flag = Cin XOR C[3]

    // Calculate the sum output
    assign Y = p ^ C[3:0];

endmodule