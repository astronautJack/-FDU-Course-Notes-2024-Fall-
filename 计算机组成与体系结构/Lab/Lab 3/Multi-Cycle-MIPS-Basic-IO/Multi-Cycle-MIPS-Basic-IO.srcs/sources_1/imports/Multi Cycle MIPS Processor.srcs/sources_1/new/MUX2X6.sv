`timescale 1ns / 1ps
// Create Date: 2024/05/10 11:58:41
// Module Name: MUX2X6
// Description: 6Î»2Â·Ñ¡ÔñÆ÷
module MUX2X6(input logic [5:0]  A0, A1,
              input logic        S,
              output logic [5:0] Y);
    
    always_comb begin
        case (S)
            1'b0: Y = A0; // Y = (S) ? A1 : A0;
            1'b1: Y = A1;
        endcase
    end

endmodule
