`timescale 1ns / 1ps
// Create Date: 2023/04/30 17:58:08
// Module Name: Decoder
// Description: 译码器
module Decoder(input  logic clk,
               input  logic Reset,
               input  logic [5:0] op,funct,
               output logic       ExtOp,PCWriteEn,MemWriteEn,IRWriteEn,
               output logic       RegWriteEn,IorD,MemToReg,RegDst,
               output logic [1:0] ALUsrcA,ALUsrcB,PCsrc,PCWrCond,
               output logic [3:0] ALUctr);
              
    localparam IFetch      = 5'b00000;//State 0,取指令+PC
    localparam ID_RFetch   = 5'b00001;//State 1,译码
    localparam MemAddr     = 5'b00010;//State 2,求存储器地址
    localparam MemFetch    = 5'b00011;//State 3,读存储器
    localparam LwFinish    = 5'b00100;//State 4,写寄存器文件
    localparam SwFinish    = 5'b00101;//State 5,写存储器
    localparam RExecute    = 5'b00110;//State 6,R型计算
    localparam RFinish     = 5'b00111;//State 7,R型写寄存器文件
    localparam BeqFinish   = 5'b01000;//State 8,beq
    localparam BneFinish   = 5'b01001;//State 9,bne
    localparam JumpFinish  = 5'b01010;//State 10,jump
    localparam AddiExecute = 5'b01011;//State 11,addi计算
    localparam OriExecute  = 5'b01100;//State 12,ori计算
    localparam AndiExecute = 5'b01101;//State 13,andi计算
    localparam AddiFinish  = 5'b01110;//State 14,addi写寄存器文件
    localparam OriFinish   = 5'b01111;//State 15,ori写寄存器文件
    localparam AndiFinish  = 5'b10000;//State 16,andi写寄存器文件
    
    localparam lw          = 6'b100011;//Opcode for lw
    localparam sw          = 6'b101011;//Opcode for sw
    localparam R           = 6'b000000;//Opcode for R-type
    localparam beq         = 6'b000100;//Opcode for beq
    localparam bne         = 6'b000101;//Opcode for bne
    localparam j           = 6'b000010;//Opcode for j
    localparam addi        = 6'b001000;//Opcode for addi
    localparam ori         = 6'b001101;//Opcode for ori
    localparam andi        = 6'b001100;//Opcode for andi
    
    logic [4:0]  CurrentState,NextState;

    //Current State Logic
    always_ff @(posedge clk or posedge Reset)
        if (Reset) CurrentState <= IFetch;
        else       CurrentState <= NextState;

    //Next State Logic
    always_comb
        case(CurrentState)
            IFetch:                    NextState = ID_RFetch;
            ID_RFetch:    case(op)
                              lw:      NextState = MemAddr;
                              sw:      NextState = MemAddr;
                              R:       NextState = RExecute;
                              beq:     NextState = BeqFinish;
                              bne:     NextState = BneFinish;
                              j:       NextState = JumpFinish;
                              addi:    NextState = AddiExecute;
                              ori:     NextState = OriExecute;
                              andi:    NextState = AndiExecute;
                              default: NextState = 5'bxxxxx;//Never happen
                          endcase
            MemAddr:      case(op)
                              lw:      NextState = MemFetch;
                              sw:      NextState = SwFinish;
                              default: NextState = 5'bxxxxx;//Safty Measure
                          endcase
            MemFetch:                  NextState = LwFinish;
            LwFinish:                  NextState = IFetch;//Next loop
            SwFinish:                  NextState = IFetch;//Next loop
            RExecute:                  NextState = RFinish;
            RFinish:                   NextState = IFetch;//Next loop
            BeqFinish:                 NextState = IFetch;//Next loop
            BneFinish:                 NextState = IFetch;//Next loop
            JumpFinish:                NextState = IFetch;//Next loop
            AddiExecute:               NextState = AddiFinish;
            OriExecute:                NextState = OriFinish;
            AndiExecute:               NextState = AndiFinish;
            AddiFinish:                NextState = IFetch;//Next loop
            OriFinish:                 NextState = IFetch;//Next loop
            AndiFinish:                NextState = IFetch;//Next loop
            default:                   NextState = 5'bxxxxx;//Safty Measure
        endcase
    //Control Code Logic
    logic        RExecuteState;
    always_comb begin
        case(CurrentState)
            RExecute:       RExecuteState = 1;
            default:        RExecuteState = 0;
        endcase
    end
    
    logic [3:0]  ALUctr0,ALUctr1;
    logic [1:0]  ALUsrcA0,ALUsrcA1;
    logic [5:0]  ControlsRenew,ControlsRenew1;
    assign {ALUsrcA,ALUctr} = ControlsRenew;
    assign {ALUsrcA1,ALUctr1} = ControlsRenew1;
    logic [19:0] Controls;
    assign {ExtOp,PCWriteEn,MemWriteEn,IRWriteEn,
            RegWriteEn,IorD,MemToReg,RegDst,
            ALUsrcA0,ALUsrcB,
            PCsrc,PCWrCond,
            ALUctr0} = Controls;
    always_comb begin
        case(CurrentState)
            IFetch:         Controls = 20'h5010f;
            ID_RFetch:      Controls = 20'h803cf;//投机计算分支目标地址
            MemAddr:        Controls = 20'h846cf;
            MemFetch:       Controls = 20'h866cf;
            LwFinish:       Controls = 20'h8e6cf;
            SwFinish:       Controls = 20'ha46cf;
            RExecute:       Controls = 20'h014c0;
            RFinish:        Controls = 20'h094c0;
            BeqFinish:      Controls = 20'hc046e;
            BneFinish:      Controls = 20'hc047e;
            JumpFinish:     Controls = 20'h4008f;
            AddiExecute:    Controls = 20'h806c0;
            OriExecute:     Controls = 20'h006c2;
            AndiExecute:    Controls = 20'h006c1;
            AddiFinish:     Controls = 20'h886c0;
            OriFinish:      Controls = 20'h086c2;
            AndiFinish:     Controls = 20'h086c1;
            default:        Controls = 20'hxxxxx;//Never happen
        endcase
    end
    
    always_comb begin
        case(funct)
            6'b100000:    ControlsRenew1 = 6'b010000;//add
            6'b100010:    ControlsRenew1 = 6'b011000;//sub
            6'b100100:    ControlsRenew1 = 6'b010001;//and
            6'b100101:    ControlsRenew1 = 6'b010010;//or
            6'b100111:    ControlsRenew1 = 6'b010011;//nor
            6'b101010:    ControlsRenew1 = 6'b011101;//slt
            6'b101011:    ControlsRenew1 = 6'b010101;//sltu
            6'b000000:    ControlsRenew1 = 6'b100100;//sll//注意三条移位指令
            6'b000010:    ControlsRenew1 = 6'b100110;//srl//ALUsrcA <= 10
            6'b000011:    ControlsRenew1 = 6'b100111;//sra
            default:      ControlsRenew1 = 6'bxxxxxx;
        endcase
    end
    
    MUX2X6 get_ALUctr({ALUsrcA0,ALUctr0},ControlsRenew1,RExecuteState,ControlsRenew);
endmodule
