`timescale 1ns / 1ps
// Create Date: 2024/06/02 09:34:39
// Design Name: MemoryDecoder
// Description: 存储器译码器

module MemoryDecoder(
	input  logic        clk,
    input  logic        Write,IorD,
    input  logic [31:0] MemAddr,
    input  logic [31:0] WriteData,
    output logic [31:0] ReadData,
    input  logic        reset,
    input  logic        buttonL,
    input  logic        buttonR,
    input  logic [15:0] switch,
    output logic [7:0]  AN,
    output logic [6:0]  A2G
);
    // 取 MemAddr 的末尾8位，用于判断是否是为 I/O 预留的地址
    logic [7:0]  addr;
    assign addr = MemAddr[7:0];
    
    logic [31:0] MemReadData;
    // 写入数据存储器的开关
    logic        MemWriteEn;
    assign MemWriteEn = Write & (addr[7] == 1'b0);
    
    Memory_IO Memory_IO(.clk(clk),
                        .MemWriteEn(MemWriteEn),
                        .MemAddr(MemAddr),
                        .MemWriteData(WriteData),
                        .MemReadData(MemReadData));
    
    logic [31:0] pReadData;
    logic [11:0] led;
    // Whether reading from IO is enabled (and IorD == 1, meaning it's not accessing an instruction)
    logic 		 pRead;
    assign pRead = (addr[7] == 1'b1 && IorD) ? 1 : 0; // 0x80
    // Whether writing to IO is enabled (and IorD == 1, meaning it's not accessing an instruction)
    logic        pWrite;
    assign pWrite = (addr[7] == 1'b1 && IorD) ? Write : 0;
    IO io(.clk(clk),
          .reset(reset),
          .pRead(addr[7]),
          .pWrite(pWrite),
          .addr(addr[3:2]),
          .pWriteData(WriteData[11:0]),
          .pReadData(pReadData),
          .buttonL(buttonL),
          .buttonR(buttonR),
          .switch(switch),
          .led(led));
    
    logic DP;
    logic [31:0] digit;
    assign digit = {switch,4'b0000,led};
    MUX7SEG mux7seg(.clk(clk),
                    .reset(reset),
                    .digit(digit),
                    .AN(AN),
                    .DP(DP),
                    .A2G(A2G));
 
    logic ReadData_select;
    assign ReadData_select = (addr[7] && IorD) ? 1 : 0;
    MUX2X32 select_ReadData(.A0(MemReadData),
                            .A1(pReadData),
                            .S(ReadData_select),
                            .Y(ReadData));
endmodule
