module top(
    input                               clk,                    // 时钟信号
    input                               rst,                    // 复位信号

    input       [31:0]                  A,                      // 输入A
    input       [31:0]                  B,                      // 输入B
    output reg  [31:0]                  C                       // 输入C
);
    assign C = A + B;

endmodule