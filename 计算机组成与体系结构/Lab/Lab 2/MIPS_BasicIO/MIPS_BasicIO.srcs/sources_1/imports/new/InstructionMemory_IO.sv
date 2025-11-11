`timescale 1ns / 1ps
// Engineer: Óº´ÞÑï
// Create Date: 2024/04/09 21:09:31
// Module Name: InstructionMemory
// Description: Ö¸Áî´æ´¢Æ÷

module InstructionMemory_IO(input  logic [5:0]  Addr,
    					 output logic [31:0] Instruction);

    /* We construct a 64x32 ROM array (256B),
       which is sufficient for storing our test instructions during simulation */
    logic [31:0] ROM[63:0];

    // Initialize memory with data from "ProgramFile.dat"
    initial begin
        $readmemh("TestIO.dat", ROM);
    end

    // Assign the instruction at the specified address to the output
    assign Instruction = ROM[Addr];

endmodule
