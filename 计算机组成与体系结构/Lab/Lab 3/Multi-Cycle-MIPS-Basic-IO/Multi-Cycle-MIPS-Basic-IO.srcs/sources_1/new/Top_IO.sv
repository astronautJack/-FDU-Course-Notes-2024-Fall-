`timescale 1ns / 1ps
// Create Date: 2024/06/02 09:54:16
// Module Name: Top_IO
// Description: 添加 IO 的终端

module Top_IO(
	input  logic clk,           // CLK100MHZ
    input  logic reset,         // BTNC
    input  logic buttonL,       // BTNL
    input  logic buttonR,       // BTNR
    input  logic [15:0] switch, // a: switch[15:8], b: switch[7:0]
    output logic [7:0]  AN,
    output logic [6:0]  A2G
);
    logic Write;  // 写信号，可能是 MemWriteEn，也可能是 ioWriteEn    
    logic IorD;
    logic [31:0] ReadData;
    logic [31:0] MemAddr,WriteData,PC;  
                         
    MIPS mips(.clk(clk),
              .Reset(reset),
              .MemReadData(ReadData),
              .MemWriteEn(Write),               // output
              .IorD(IorD),                      // output
              .MemAddr(MemAddr),		        // output
              .MemWriteData(WriteData),		    // output
              .PC(PC));	                        // output
    
    MemoryDecoder MemoryDecoder(.clk(clk),
                                .Write(Write),
                                .IorD(IorD),
                                .MemAddr(MemAddr),
                                .WriteData(WriteData),
                                .ReadData(ReadData),	// Output
                                .reset(reset),
                                .buttonL(buttonL),
                                .buttonR(buttonR),
                                .switch(switch),
                                .AN(AN),				// Output
                                .A2G(A2G));				// Output
endmodule
