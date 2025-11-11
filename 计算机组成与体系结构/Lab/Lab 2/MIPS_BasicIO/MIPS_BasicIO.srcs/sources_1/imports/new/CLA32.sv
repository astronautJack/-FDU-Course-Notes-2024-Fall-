`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/04/10 20:51:11
// Design Name: 
// Module Name: CLA32
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


module CLA32(
    input  logic [31:0] A, B,     		// Input operands A and B
    input         		Cin,      		// Carry-in
    output logic [31:0] Y,        		// Sum output
    output              OF, SF, ZF, CF    
	// Overflow flag, Sign flag, Zero flag, Carry flag
);

    // Wires to store the carry-out of each CLA4 block
    logic [7:0] Cout_set;

    // Wire to store the result of B XOR Cin
    logic [31:0] Bxor;
    
    // Calculate B XOR Cin
    assign Bxor = B ^ {32{Cin}};

    /* Instantiate 8 CLA4 blocks to calculate carry-out for each set of 4 bits
       Overflow flag is already provided by CLA4_Marked		*/
    CLA4 add0(A[3:0], Bxor[3:0], Cin, Cout_set[0], Y[3:0]);
    CLA4 add1(A[7:4], Bxor[7:4], Cout_set[0], Cout_set[1], Y[7:4]);
    CLA4 add2(A[11:8], Bxor[11:8], Cout_set[1], Cout_set[2], Y[11:8]);
    CLA4 add3(A[15:12], Bxor[15:12], Cout_set[2], Cout_set[3], Y[15:12]);
    CLA4 add4(A[19:16], Bxor[19:16], Cout_set[3], Cout_set[4], Y[19:16]);
    CLA4 add5(A[23:20], Bxor[23:20], Cout_set[4], Cout_set[5], Y[23:20]);
    CLA4 add6(A[27:24], Bxor[27:24], Cout_set[5], Cout_set[6], Y[27:24]);
    CLA4_Marked add7(A[31:28], Bxor[31:28], Cout_set[6], Cout_set[7], Y[31:28], OF);
    
    // Calculate carry-out for the entire 32-bit adder
    logic Cout = Cout_set[7]; // Highest carry-out

    // Calculate sign flag
    assign SF = Y[31];

    // Calculate zero flag
    assign ZF = (Y) ? 0 : 1;

    // Calculate carry flag
    assign CF = Cout ^ Cin;

endmodule
