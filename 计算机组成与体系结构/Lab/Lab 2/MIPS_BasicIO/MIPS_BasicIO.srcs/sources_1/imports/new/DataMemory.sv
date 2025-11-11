`timescale 1ns / 1ps
// Engineer: Óº´ÞÑï
// Create Date: 2024/04/09 21:09:27
// Module Name: DataMemory
// Description: Êý¾Ý´æ´¢Æ÷

module DataMemory(input               clk, DMWriteEn,
    			  input        [31:0] DMDataAddr, DMWriteData,
    			  output logic [31:0] DMReadData);
    /* Although ideally, we would construct a 4GB DataMemory with:
           logic [31:0] RAM[2**30-1:0];
       it would be computationally expensive during simulation.
       Therefore, we have opted for a smaller 1024B DataMemory.
       Additionally, during simulation, we ensure that no data is stored
       in addresses higher than 0x00000400. */
    logic [31:0] RAM[2**8-1:0];
    
    // Word Aligned
    assign DMReadData = RAM[DMDataAddr[31:2]];
    
    always_ff @(posedge clk)
        if (DMWriteEn) 
            RAM[DMDataAddr[31:2]] <= DMWriteData;

endmodule