# 实验代码说明

1. 单周期 MIPS 处理器封装在 `Top.sv` 中，  
   对应的仿真 (simulation) 代码为 `test_Top.sv`.  
   指令存储器 `InstructionMemory.sv` 中的 `$readmemh` 语句可以指定使用哪个 `.dat` 文件中的机器码，  
   有两种选择：
   - `ProgramFile.dat` 是教材《数字设计与计算机体系结构》(D.M.Harris 2nd Edition) 中文译本 P276 给出的 MIPS 汇编代码对应的机器码；
   - `ProgramFile2.dat` 是我编写的 MIPS 汇编代码对应的机器码，  
     用于测试 `addi`、`ori`、`bne` 等额外实现的指令；
2. 添加了 I/O 模块的单周期 MIPS 处理器封装在 `Top_IO.sv` 中，  
   对应的仿真代码为 `test_Top_IO.sv` (用于检验数据存储器译码器 `DMDecoder.sv` 是否正确)  
   指令存储器 `InstructionMemory_IO.sv` 中的 `$readmemh` 语句指定使用 `TestIO.dat` 文件中的机器码.
3. 为了进行上板验证，  
   我额外设计了时钟分频模块 `CLK_div.sv`，封装进 `Top_NEXYS_A7_100T.sv` 中.  
   对应的仿真代码为 `test_Top_NEXYS_A7_100T.sv` (用于检验时钟分频模块是否正确)  
   约束文件为 `Nexys-A7-100T-Master.xdc`