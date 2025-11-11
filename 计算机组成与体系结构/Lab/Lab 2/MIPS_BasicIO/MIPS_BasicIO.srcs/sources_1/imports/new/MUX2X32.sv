module MUX2X32(input  logic [31:0]  A0, A1,
    		   input              	S,
    		   output logic [31:0] 	Y);

    // Combinational logic to select between A0 and A1 based on S
    always_comb begin
        case (S)
            1'b0: Y = A0;  // When S is 0, Y is assigned A0
            1'b1: Y = A1;  // When S is 1, Y is assigned A1
            default: Y = 32'b0;
        endcase
    end

endmodule