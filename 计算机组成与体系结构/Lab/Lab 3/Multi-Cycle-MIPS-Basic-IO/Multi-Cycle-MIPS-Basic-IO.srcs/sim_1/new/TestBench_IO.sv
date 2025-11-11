`timescale 1ns / 1ps
// Create Date: 2024/06/02 09:54:16
// Module Name: TestBench_IO
// Description: ÃÌº” IO µƒ≤‚ ‘Ã®

module TestBench_IO();
    logic 		 clk;	  // CLK100MHZ
    logic 		 reset;   // BTNC
    logic 		 buttonL; // BTNL
    logic 		 buttonR; // BTNR
    logic [15:0] switch;  // SW
    logic [7:0]  AN;
    logic [6:0]  A2G;
    
    // instantiate device to be tested
    Top_IO top_IO(.clk(clk),
                  .reset(reset),
                  .buttonL(buttonL),
                  .buttonR(buttonR),
                  .switch(switch),
                  .AN(AN),
                  .A2G(A2G));
    
    // initialize test
    initial begin
        #0; reset <= 1;
        #2; reset <= 0;
        #2; buttonL <= 1; buttonR <= 1;
        #2; switch 	<= 16'b00000100_00001000; // 4 + 8
    end
    
    // generate clock to sequence tests
    always begin
        clk <= 1; # 5;
        clk <= 0; # 5;
    end

endmodule
