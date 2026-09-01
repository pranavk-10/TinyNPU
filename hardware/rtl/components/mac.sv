// ============================================================
// TinyNPU - MAC Unit
// ============================================================
//
// MAC = Multiply Accumulate
//
// This module calculates:
//
//     result = bias + SUM(input[i] * weight[i])
//
// Inputs:
//     input  = signed INT8
//     weight = signed INT8
//
// Product:
//     INT8 × INT8 → INT16
//
// Accumulator:
//     multiple products are accumulated into INT32
//
// This module is combinational.
// ============================================================


module mac #(

    // N tells the MAC how many input/weight pairs it receives.
    // For layer 1, N = 21.
    // For layer 2, N = 14.
    // For layer 3, N = 7.
    parameter integer N = 21

)(

    // Array containing N signed 8-bit input values.
    input logic signed [7:0] inputs [0:N-1],

    // Array containing N signed 8-bit weights.
    input logic signed [7:0] weights [0:N-1],

    // Bias is already represented as INT32.
    input logic signed [31:0] bias,

    // Final accumulated result.
    output logic signed [31:0] result

);


    // Integer used as the loop counter.
    integer i;


    // 32-bit accumulator.
    //
    // This is where all multiplication results are added.
    logic signed [31:0] accumulator;


    // A multiplication of two signed INT8 values
    // requires up to 16 bits.
    logic signed [15:0] product;


    // always_comb means this logic continuously responds
    // to changes in inputs, weights, or bias.
    always_comb begin


        // Start the accumulator with the bias.
        accumulator = bias;


        // Loop through every input/weight pair.
        //
        // For N = 21:
        //
        // i = 0
        // i = 1
        // ...
        // i = 20
        //
        // Every iteration performs:
        //
        // input[i] × weight[i]
        //
        // and adds that product to accumulator.
        for (i = 0; i < N; i = i + 1) begin


            // Multiply the signed INT8 input
            // by the signed INT8 weight.
            //
            // 8-bit × 8-bit = 16-bit.
            product = inputs[i] * weights[i];


            // Add the product to the 32-bit accumulator.
            accumulator = accumulator + product;


        end


        // After all N products have been accumulated,
        // send the final value to the output.
        result = accumulator;


    end


endmodule