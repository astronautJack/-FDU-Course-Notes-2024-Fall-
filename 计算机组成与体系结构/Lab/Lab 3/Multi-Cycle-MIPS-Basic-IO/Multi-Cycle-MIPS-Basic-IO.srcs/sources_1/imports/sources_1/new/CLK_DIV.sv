`timescale 1ns / 1ps
// Create Date: 2024/04/16 12:45:44
// Module Name: CLK_DIV
// Description: 

module CLK_div #(parameter int N = 10000)  // 使用int指定参数类型
(
    input logic CLK_in,
    output logic CLK_out
);

    logic [31:0] counter = 0;
    logic out = 0;

    // 使用单个always_ff块处理计数和输出翻转
    always_ff @(posedge CLK_in) begin
        if (counter == N) begin
            counter <= 0;     // 计数器清零
            out <= ~out;      // 输出信号翻转
        end else begin
            counter <= counter + 1; // 计数器递增
        end
    end

    assign CLK_out = out;  // 将内部信号out赋值给输出CLK_out

endmodule