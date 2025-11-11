`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/04/15 16:40:03
// Design Name: 
// Module Name: DMDecoder
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


module DMDecoder(
	input  logic        clk,
    input  logic        Write,
    input  logic [7:0]  addr,
    input  logic [31:0] WriteData,
    output logic [31:0] ReadData,
    input  logic        reset,
    input  logic        buttonL,
    input  logic        buttonR,
    input  logic [15:0] switch,
    output logic [7:0]  AN,
    output logic [6:0]  A2G
);
    
    logic [31:0] DMReadData;
    // 写入数据存储器的开关
    logic        DMWriteEn;
    assign DMWriteEn = Write & (addr[7] == 1'b0);
    
    DataMemory DM(.clk(clk),
                  .DMWriteEn(DMWriteEn),
                  .DMDataAddr({24'b0,addr}),
                  .DMWriteData(WriteData),
                  .DMReadData(DMReadData));
    
    logic [31:0] pReadData;
    logic [11:0] led;
    // Whether reading from IO is enabled.
    logic 		 pRead;
    assign pRead = (addr[7] == 1'b1) ? 1 : 0; // 0x80
    // Whether writing to IO is enabled.
    logic        pWrite;
    assign pWrite = (addr[7] == 1'b1) ? Write : 0;
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
    MUX7SEG mux7seg(.clk(clk),
                    .reset(reset),
                    .digit({switch,4'b0000,led}),
                    .AN(AN),
                    .DP(DP),
                    .A2G(A2G));
    
    MUX2X32 select_ReadData(.A0(DMReadData),
                            .A1(pReadData),
                            .S(addr[7]),
                            .Y(ReadData));
endmodule
