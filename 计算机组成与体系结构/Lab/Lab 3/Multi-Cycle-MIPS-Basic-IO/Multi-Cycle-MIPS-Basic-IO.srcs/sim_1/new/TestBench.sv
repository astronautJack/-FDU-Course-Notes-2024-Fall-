`timescale 1ns / 1ps
// Create Date: 2023/05/06 15:10:49
// Module Name: TestBench
// Description: ∑¬’Ê≤‚ ‘
module TestBench();
    logic        clk,Reset,MemWriteEn,IorD;
    logic [31:0] MemAddr,MemWriteData,PC,MemReadData;
    Top top(clk,Reset,MemWriteEn,IorD,MemAddr,MemWriteData,PC,MemReadData);
    initial begin//initialize test
        Reset <= 1;
        # 2;
        Reset <= 0;
    end
    always begin//generate clock
        clk <= 1;
        # 5;
        clk <= 0;
        # 5;
    end
    always @(negedge clk)//check results
    begin
        if(MemWriteEn) begin
            if(MemAddr === 84 & MemWriteData === 7) begin
                $display("Simulation succeeded");
                $stop;
            end else if(MemAddr !== 80) begin
                $display("Simulation failed");
                $stop;
            end
        end
    end
endmodule
