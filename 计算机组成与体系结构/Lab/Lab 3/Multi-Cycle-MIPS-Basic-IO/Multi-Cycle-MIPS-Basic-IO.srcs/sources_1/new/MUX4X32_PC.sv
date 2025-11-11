`timescale 1ns / 1ps
// Create Date: 2024/04/30 12:31:41
// Module Name: MUX4X32
// Description: 32Î»4Â·Ñ¡ÔñÆ÷
module MUX4X32_PC(
               input logic [31:0] PCadd4, BranchPC, JumpPC, PC,
               input logic        Branch,
               input logic [1:0]  PCsrc,
               output logic [31:0] NextPC);

    always_comb begin
        case(PCsrc)
            2'b01: 
                case(Branch)
                    1'b1: NextPC = BranchPC;
                    1'b0: NextPC = PC;
                endcase
            2'b10:        NextPC = JumpPC;
            2'b00:        NextPC = PCadd4;
            2'b11:        NextPC = PC;
            default:      NextPC = PC;
        endcase
    end
    
endmodule
