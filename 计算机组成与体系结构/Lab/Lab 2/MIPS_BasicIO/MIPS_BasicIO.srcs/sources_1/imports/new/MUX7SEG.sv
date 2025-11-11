`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/04/15 16:43:18
// Design Name: 
// Module Name: MUX7SEG
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

// 定义符号"="的段编码
localparam SEG_EQUAL = 7'b011_0111;  // 37

module MUX7SEG(
    input  logic clk, reset,
    input  logic [31:0] digit, // {switch(16bit), 4'b0000, led(12bit)}
    output logic [7:0] AN,     // 用来控制哪个数码管被激活
    output logic DP,
    output logic [6:0] A2G     // 用来控制数码管上的七个段，以显示相应的数字
);

    // 当前显示的数字
    logic [3:0] current_digit;

    // 显示刷新计数器
    logic [2:0] refresh_counter;
    
    // 数字转换为7段信号
    always_ff @(posedge clk or posedge reset) begin
    	if (reset) begin
            current_digit <= 0;
            refresh_counter <= 0;
            AN <= 8'b1111_1111;     // 关闭所有显示
            DP <= 1;                // 关闭小数点
    	end 
        else begin
            current_digit <= digit[4*refresh_counter +: 4];  // 选择要显示的数字部分
            AN <= ~(1'b1 << refresh_counter);  				 // 激活当前数码管
            refresh_counter <= (refresh_counter + 1'b1) % 8; // 循环显示，确保在0-7之间
    	end
	end
    
    // 解码当前数字到7段显示
    always_comb begin
        if (refresh_counter == 4) begin // 第四个显示总是显示"="
            A2G = SEG_EQUAL;
        end
        else begin
            case (current_digit)
                4'd0:  A2G = 7'b100_0000; // SEG_0 = 40
                4'd1:  A2G = 7'b111_1001; // SEG_1 = 79
                4'd2:  A2G = 7'b010_0100; // SEG_2 = 24
                4'd3:  A2G = 7'b011_0000; // SEG_3 = 30
                4'd4:  A2G = 7'b001_1001; // SEG_4 = 19
                4'd5:  A2G = 7'b001_0010; // SEG_5 = 12
                4'd6:  A2G = 7'b000_0010; // SEG_6 = 02
                4'd7:  A2G = 7'b111_1000; // SEG_7 = 78
                4'd8:  A2G = 7'b000_0000; // SEG_8 = 00
                4'd9:  A2G = 7'b001_0000; // SEG_9 = 10
                4'd10: A2G = 7'b000_1000; // SEG_A = 08
                4'd11: A2G = 7'b000_0011; // SEG_B = 03
                4'd12: A2G = 7'b100_0110; // SEG_C = 46
                4'd13: A2G = 7'b010_0001; // SEG_D = 21
                4'd14: A2G = 7'b000_0110; // SEG_E = 06
                4'd15: A2G = 7'b000_1110; // SEG_F = 0d
                default: A2G = 7'b111_1111; // Invalid digit = 7f
            endcase
        end
    end
endmodule