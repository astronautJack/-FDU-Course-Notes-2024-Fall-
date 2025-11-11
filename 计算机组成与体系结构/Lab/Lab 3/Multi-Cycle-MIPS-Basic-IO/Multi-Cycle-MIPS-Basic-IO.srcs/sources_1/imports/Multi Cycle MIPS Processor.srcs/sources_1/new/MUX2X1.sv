`timescale 1ns / 1ps
// Create Date: 2024/05/10 12:33:29
// Module Name: MUX2X1
// Description: 1Î»2Â·Ñ¡ÔñÆ÷
module MUX2X1(input logic  A0, A1,
              input logic  S,
              output logic Y);

    always_comb
    begin
        case(S)
            1'b0: Y <= A0; // Y = (S) ? A1 : A0;
            1'b1: Y <= A1;
        endcase
    end

endmodule
