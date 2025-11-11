`timescale 1ns / 1ps
// Create Date: 2024/04/30 12:31:41
// Module Name: MUX4X32
// Description: 32Î»4Â·Ñ¡ÔñÆ÷
module MUX3X32_Data(
               input logic [31:0] Data0, Data1, Data2,
               input logic [1:0]  SourceControl,
               output logic [31:0] Data);

    always_comb begin
        case(SourceControl)
            2'b00: Data = Data0;
            2'b01: Data = Data1;
            2'b10: Data = Data2;
            default: Data = Data0;
        endcase
    end
    
endmodule
