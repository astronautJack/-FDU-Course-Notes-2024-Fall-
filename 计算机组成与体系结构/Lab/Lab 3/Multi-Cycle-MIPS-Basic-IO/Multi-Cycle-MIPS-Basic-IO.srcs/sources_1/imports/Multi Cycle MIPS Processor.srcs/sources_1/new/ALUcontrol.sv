`timescale 1ns / 1ps
// Create Date: 2024/04/19 10:05:09
// Module Name: ALUcontrol
// Description: ALU¿ØÖÆÆ÷
module ALUcontrol(input  logic [3:0] ALUctr,
                  output logic       SUBctr,OFctr,SIGctr,
                  output logic [2:0] OPctr);
    logic [5:0] Controls;
    assign {SUBctr,OFctr,SIGctr,OPctr} = Controls;
    always_comb begin
        case(ALUctr)
            4'b0000:Controls = 6'b010000;//add\addi
            4'b1000:Controls = 6'b110000;//sub\subi
            4'b0001:Controls = 6'b000001;//and\andi
            4'b0010:Controls = 6'b000010;//or\ori
            4'b0011:Controls = 6'b000011;//nor\nori
            4'b0101:Controls = 6'b100101;//sltu
            4'b1101:Controls = 6'b101101;//slt\slti
            4'b0100:Controls = 6'b000100;//sll
            4'b0110:Controls = 6'b000110;//srl
            4'b0111:Controls = 6'b000111;//sra
            4'b1110:Controls = 6'b100000;//beq\bne(subu)
            4'b1111:Controls = 6'b000000;//lw\sw(addu)
            default: Controls = 6'b000000; // default case to avoid latches
        endcase
    end
endmodule