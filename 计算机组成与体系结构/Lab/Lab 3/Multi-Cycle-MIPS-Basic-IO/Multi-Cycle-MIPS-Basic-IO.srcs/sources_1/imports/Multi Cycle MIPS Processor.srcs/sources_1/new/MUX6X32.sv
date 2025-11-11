`timescale 1ns / 1ps
// Create Date: 2024/04/12 12:31:41
// Module Name: MUX6X32
// Description: 32Î»6Â·Ñ¡ÔñÆ÷
module MUX6X32(input logic [31:0] Y_addsub, Y_and, Y_or, Y_nor, Y_slt, Y_shift,
               input logic [2:0]  OPctr,
               output logic [31:0] Y);

    always_comb
    begin
        case(OPctr)
            3'b000: Y <= Y_addsub; // add\addi\sub\subi\lw\sw\beq\bne\slt
            3'b001: Y <= Y_and;    // and\andi
            3'b010: Y <= Y_or;     // or\ori
            3'b011: Y <= Y_nor;    // nor\nori
            3'b101: Y <= Y_slt;    // slt\slti¡¢sltu
            3'b100: Y <= Y_shift;  // Âß¼­×óÒÆsll
            3'b110: Y <= Y_shift;  // Âß¼­ÓÒÒÆsrl
            3'b111: Y <= Y_shift;  // ËãÊõÓÒÒÆsra
        endcase
    end

endmodule
