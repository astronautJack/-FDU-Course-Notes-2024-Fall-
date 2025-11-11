`timescale 1ns / 1ps
// Create Date: 2024/04/30 15:56:08
// Design Name: Memory_IO
// Description: 测试 IO 的存储器
module Memory_IO(input logic            clk, MemWriteEn,
                 input logic [31:0]     MemAddr, MemWriteData,
                 output logic [31:0]    MemReadData);
    
    /* Although ideally, we would construct a 4GB DataMemory with:
           logic [31:0] RAM[2**30-1:0];
       it would be computationally expensive during simulation.
       Therefore, we have opted for a smaller 1024B = 1KB DataMemory.
       Additionally, during simulation, we ensure that no data is stored
       in addresses higher than 0x00000400. 						  */
    
    logic [31:0] memory[2**8-1:0];
    
    initial // 初始化指令
    begin
        $readmemh("TestIO.dat", memory, 2**7);
    end
    
    always_comb // 读
    begin
        if (MemAddr[31:2] < 2**7)
            MemReadData = memory[MemAddr[9:2]]; // 读数据
        else
            MemReadData = memory[((MemAddr - 32'h00400000) >> 2) + 2**7]; // 读指令
    end
    
    always_ff @(posedge clk) // 写
    begin
        if (MemWriteEn) 
            memory[MemAddr[9:2]] <= MemWriteData; // 写数据
    end
    
endmodule