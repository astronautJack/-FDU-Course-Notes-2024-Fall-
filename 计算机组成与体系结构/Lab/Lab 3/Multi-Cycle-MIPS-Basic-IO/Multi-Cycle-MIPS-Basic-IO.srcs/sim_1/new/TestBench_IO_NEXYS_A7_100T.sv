`timescale 1ns / 1ps
// Create Date: 2024/06/02 09:54:16
// Module Name: TestBench_IO_NEXYS_A7_100T
// Description: 添加 IO 的测试台 (加入时钟分频模块以适配实验板)

module TestBench_IO_NEXYS_A7_100T();
    logic 		 clk;	  // CLK100MHZ
    logic 		 reset;   // BTNC
    logic 		 buttonL; // BTNL
    logic 		 buttonR; // BTNR
    logic [15:0] switch;  // SW
    logic [7:0]  AN;
    logic [6:0]  A2G;
    
    // instantiate device to be tested
    Top_IO_NEXYS_A7_100T top_nexys(
                  .clk(clk),
                  .reset(reset),
                  .buttonL(buttonL),
                  .buttonR(buttonR),
                  .switch(switch),
                  .AN(AN),
                  .A2G(A2G));
    
    // initialize test
    initial begin
        #0; reset <= 1;
        #5; reset <= 0;
        #5; buttonL <= 1; buttonR <= 1;
        #5; switch 	<= 16'b00000100_00001000; // 4 + 8
    end
    
    // generate clock to sequence tests
    always begin
        clk <= 1; # 5;
        clk <= 0; # 5;
    end

endmodule