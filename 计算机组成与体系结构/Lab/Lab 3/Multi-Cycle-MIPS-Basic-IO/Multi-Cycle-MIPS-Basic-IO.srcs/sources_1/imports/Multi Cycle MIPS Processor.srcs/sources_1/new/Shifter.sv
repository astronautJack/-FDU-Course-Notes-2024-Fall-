`timescale 1ns / 1ps
// Create Date: 2023/04/12 08:55:39
// Module Name: Shifter
// Description: 32位移位器
module Shifter(input logic [31:0] A,
               input logic [4:0]  SA,//SA for Shift Amount,5位移位数
               input logic        Right,Arith,//用于判别是右移还是左移;用于判别是算术移位还是逻辑移位
               output logic [31:0] Y);
    logic [31:0] t0,t1,t2,t3,t4,s0,s1,s2,s3,s4;//临时变量
    logic [31:0] l0,l1,l2,l3,l4,r0,r1,r2,r3,r4;
    logic [31:0] right_shift;
    logic indicator;
    logic [15:0] left_shift;

    assign left_shift = 16'b0;
    assign indicator = A[31] & Arith;
    assign right_shift = {16{indicator}};
    
    //4th    根据移位数的第四位SA[4]对A进行移位
    assign l4 = {A[15:0],left_shift[15:0]};
    assign r4 = {right_shift[15:0],A[15:0]};
    MUX2X32 Mux4LandR(l4,r4,Right,t4);//根据Right选择左移还是右移，存入t4
    MUX2X32 shift_on_4th(A,t4,SA[4],s4);//根据移位数的第四位SA[4]对A进行移位
    //3rd    根据移位数的第三位SA[3]对s4进行移位
    assign l3 = {s4[23:0],left_shift[7:0]};
    assign r3 = {right_shift[7:0],s4[31:8]};
    MUX2X32 Mux3LandR(l3,r3,Right,t3);//根据Right选择左移还是右移，存入t3
    MUX2X32 shift_on_3rd(s4,t3,SA[3],s3);//根据移位数的第三位SA[3]对s4进行移位
    //2nd    根据移位数的第二位SA[2]对s3进行移位
    assign l2 = {s3[27:0],left_shift[3:0]};
    assign r2 = {right_shift[3:0],s3[31:4]};
    MUX2X32 Mux2LandR(l2,r2,Right,t2);//根据Right选择左移还是右移，存入t2
    MUX2X32 shift_on_2nd(s3,t2,SA[2],s2);//根据移位数的第二位SA[2]对s3进行移位
    //1st    根据移位数的第一位SA[1]对s2进行移位
    assign l1 = {s2[29:0],left_shift[1:0]};
    assign r1 = {right_shift[1:0],s2[31:2]};
    MUX2X32 Mux1LandR(l1,r1,Right,t1);//根据Right选择左移还是右移，存入t1
    MUX2X32 shift_on_1st(s2,t1,SA[1],s1);//根据移位数的第一位SA[1]对s2进行移位    
    //0th    根据移位数的第0位SA[0]对s1进行移位
    assign l0 = {s1[30:0],left_shift[0]};
    assign r0 = {right_shift[0],s1[31:1]};
    MUX2X32 Mux0LandR(l0,r0,Right,t0);//根据Right选择左移还是右移，存入t0
    MUX2X32 shift_on_0th(s1,t0,SA[0],s0);//根据移位数的第0位SA[1]对s1进行移位   
    assign Y = s0;
endmodule
