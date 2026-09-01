// ============================================================
// TinyNPU - Single Neuron
// ============================================================
//
// Flat packed-bus implementation.
//
// N INT8 inputs
// N INT8 weights
// INT32 bias
//        ↓
// INT32 accumulator
//        ↓
// requantization
//        ↓
// INT8
//        ↓
// ReLU
//        ↓
// INT8 output
//
// ============================================================


module neuron #(

    // Number of inputs.
    parameter integer N = 21,

    // Requantization multiplier.
    parameter integer MULTIPLIER = 1223,

    // Requantization shift.
    parameter integer SHIFT = 20

)(

    // --------------------------------------------------------
    // All inputs packed into ONE bus.
    //
    // N × 8 bits.
    //
    // For N = 21:
    //
    // 21 × 8 = 168 bits.
    // --------------------------------------------------------

    input logic signed [(N*8)-1:0] inputs_bus,


    // --------------------------------------------------------
    // All weights packed into ONE bus.
    //
    // Also N × 8 bits.
    // --------------------------------------------------------

    input logic signed [(N*8)-1:0] weights_bus,


    // --------------------------------------------------------
    // INT32 bias.
    // --------------------------------------------------------

    input logic signed [31:0] bias,


    // --------------------------------------------------------
    // One INT8 output.
    // --------------------------------------------------------

    output logic signed [7:0] output_value

);


    // --------------------------------------------------------
    // INT32 accumulator.
    // --------------------------------------------------------

    logic signed [31:0] accumulator;


    // --------------------------------------------------------
    // 64-bit intermediate value.
    // --------------------------------------------------------

    logic signed [63:0] scaled_value;


    // --------------------------------------------------------
    // INT8 requantized value.
    // --------------------------------------------------------

    logic signed [7:0] quantized_value;


    // --------------------------------------------------------
    // Loop counter.
    // --------------------------------------------------------

    integer i;


    // ========================================================
    // NEURON CALCULATION
    // ========================================================

    always @(*) begin

        // ----------------------------------------------------
        // Start with bias.
        // ----------------------------------------------------

        accumulator = bias;


        // ----------------------------------------------------
        // Process every input/weight pair.
        // ----------------------------------------------------

        for (
            i = 0;
            i < N;
            i = i + 1
        ) begin


            // ------------------------------------------------
            // Extract one INT8 input.
            //
            // Input i occupies:
            //
            //     i*8 +: 8
            // ------------------------------------------------

            accumulator =
                accumulator
                +
                (
                    $signed(inputs_bus[i*8 +: 8])
                    *
                    $signed(weights_bus[i*8 +: 8])
                );


        end


        // ----------------------------------------------------
        // Apply fixed-point multiplier.
        // ----------------------------------------------------

        scaled_value =
            accumulator * MULTIPLIER;


        // ----------------------------------------------------
        // Convert fixed-point value back to integer.
        // ----------------------------------------------------

        scaled_value =
            scaled_value >>> SHIFT;


        // ----------------------------------------------------
        // Saturate to INT8 maximum.
        // ----------------------------------------------------

        if (scaled_value > 127) begin

            quantized_value = 8'sd127;

        end


        // ----------------------------------------------------
        // Saturate to INT8 minimum.
        // ----------------------------------------------------

        else if (scaled_value < -128) begin

            quantized_value = -8'sd128;

        end


        // ----------------------------------------------------
        // Normal INT8 value.
        // ----------------------------------------------------

        else begin

            quantized_value = scaled_value[7:0];

        end


        // ----------------------------------------------------
        // ReLU.
        //
        // Negative → 0
        // Positive → unchanged
        // ----------------------------------------------------

        if (quantized_value < 0) begin

            output_value = 8'sd0;

        end
        else begin

            output_value = quantized_value;

        end

    end


endmodule