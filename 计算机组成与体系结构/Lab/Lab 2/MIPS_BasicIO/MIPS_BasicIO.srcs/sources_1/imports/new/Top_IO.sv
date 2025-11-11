`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/04/15 16:38:56
// Design Name: 
// Module Name: Top_IO
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


module Top_IO(
	input  logic clk,           // CLK100MHZ
    input  logic reset,         // BTNC
    input  logic buttonL,       // BTNL
    input  logic buttonR,       // BTNR
    input  logic [15:0] switch, // a: switch[15:8], b: switch[7:0]
    output logic [7:0]  AN,
    output logic [6:0]  A2G
);
    
    logic [31:0] PC, Instruction;
    InstructionMemory_IO IM(.Addr(PC[7:2]),
                            .Instruction(Instruction));	// output
    
    // 写信号，可能是DMWriteEn，也可能是ioWriteEn
    logic Write;
    logic [31:0] ReadData;
    logic [31:0] WriteData;       
    logic [31:0] DataAddr;        
    
    MIPS mips(.clk(clk),
              .Reset(reset),
              .Instruction(Instruction),
              .DMReadData(ReadData),
              .PC(PC),					// output
              .DMWriteEn(Write),		// output
              .ALUresult(DataAddr),		// output
              .DMWriteData(WriteData));	// output
    
    DMDecoder DMdecoder(.clk(clk),
                        .Write(Write),
                        .addr(DataAddr[7:0]),
                        .WriteData(WriteData),
                        .ReadData(ReadData),	// Output
                        .reset(reset),
                        .buttonL(buttonL),
                        .buttonR(buttonR),
                        .switch(switch),
                        .AN(AN),				// Output
                        .A2G(A2G));				// Output
    
endmodule
