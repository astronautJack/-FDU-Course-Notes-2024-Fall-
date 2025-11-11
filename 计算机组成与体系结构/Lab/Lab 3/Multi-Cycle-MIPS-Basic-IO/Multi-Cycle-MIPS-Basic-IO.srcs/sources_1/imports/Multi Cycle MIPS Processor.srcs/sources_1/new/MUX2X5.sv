`timescale 1ns / 1ps
// Create Date: 2024/04/20 12:26:05
// Module Name: MUX2X5
// Description: 5Î»2Â·Ñ¡ÔñÆ÷
module MUX2X5(input logic [4:0] A0, A1,
              input logic       S,
              output logic [4:0] Y);

    always_comb begin
        case(S)
            1'b0: Y = A0; // Y = (S) ? A1 : A0;
            1'b1: Y = A1;
        endcase
    end

endmodule
