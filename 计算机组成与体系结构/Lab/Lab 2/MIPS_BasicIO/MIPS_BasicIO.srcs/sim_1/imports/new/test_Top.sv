`timescale 1ns / 1ps
// Create Date: 2024/04/15 12:10:24
// Module Name: test_Top
// Description: TopµÄ·ÂÕæ²âÊÔ

module test_Top();
    logic        clk;
    logic        Reset;
    logic [31:0] DMWriteData,DMDataAddr;
    logic        DMWriteEn;
//instantiate device to be tested
    Top top(clk,Reset,DMWriteData,DMDataAddr,DMWriteEn);
//initialize test
    initial begin
        Reset <= 1; #22; Reset <= 0;
    end
//generate clock to sequence tests
    always begin
        clk <= 1; #5 ; clk <= 0; #5;
    end
//check that 7 gets written to address 84
    always @(negedge clk) begin
        if(DMWriteEn) begin
            if(DMDataAddr === 84 & DMWriteData === 7) begin
                $display("Simulation succeeded");
                $stop;
            end
        else if(DMDataAddr !== 80) begin
            $display("Simulation failed");
            $stop;
            end
        end
    end
endmodule
