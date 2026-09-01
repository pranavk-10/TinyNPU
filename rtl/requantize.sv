// ============================================================
// TinyNPU - INT32 to INT8 Requantization
// ============================================================
//
// Purpose:
//
//     Convert the INT32 accumulator produced by the MAC
//     back into an INT8 activation.
//
//
//
// Mathematical operation:
//
//     real_value = accumulator
//                  × input_scale
//                  × weight_scale
//
//     output_int8 = real_value / activation_scale
//
// Therefore:
//
//     output_int8 = accumulator
//                  × (input_scale × weight_scale)
//                  / activation_scale
//
//
// Floating-point arithmetic is NOT used in the RTL.
//
// Instead, we approximate the scale ratio using:
//
//     output = (accumulator × MULTIPLIER) >>> SHIFT
//
// ============================================================


module requantize #(

    // --------------------------------------------------------
    // Integer approximation of the floating-point scale ratio.
    //
    // For example:
    //
    //     ratio ≈ MULTIPLIER / 2^SHIFT
    //
    // --------------------------------------------------------

    parameter integer MULTIPLIER = 1223,

    // --------------------------------------------------------
    // Number of bits to shift right.
    // --------------------------------------------------------

    parameter integer SHIFT = 20

)(

    // --------------------------------------------------------
    // INT32 accumulator produced by the MAC.
    // --------------------------------------------------------

    input logic signed [31:0] accumulator,

    // --------------------------------------------------------
    // INT8 output activation.
    // --------------------------------------------------------

    output logic signed [7:0] output_value

);


    // --------------------------------------------------------
    // 64-bit temporary value.
    //
    // We use 64 bits because:
    //
    //     INT32 × integer multiplier
    //
    // can exceed 32 bits.
    // --------------------------------------------------------

    logic signed [63:0] scaled_value;


    // --------------------------------------------------------
    // Value after the right shift.
    // --------------------------------------------------------

    logic signed [63:0] shifted_value;


    // --------------------------------------------------------
    // Combinational logic.
    //
    // There is no clock in this module.
    // The output changes whenever accumulator changes.
    // --------------------------------------------------------

    always_comb begin


        // ----------------------------------------------------
        // Multiply the INT32 accumulator by the integer
        // approximation of the scale ratio.
        // ----------------------------------------------------

        scaled_value =
            accumulator * MULTIPLIER;


        // ----------------------------------------------------
        // Perform an arithmetic right shift.
        //
        // >>> preserves the sign for signed numbers.
        //
        // This approximates division by 2^SHIFT.
        // ----------------------------------------------------

        shifted_value =
            scaled_value >>> SHIFT;


        // ----------------------------------------------------
        // INT8 maximum is +127.
        //
        // If our calculated value is larger than 127,
        // clamp it to 127.
        // ----------------------------------------------------

        if (shifted_value > 127) begin


            output_value = 8'sd127;


        end


        // ----------------------------------------------------
        // INT8 minimum is -128.
        //
        // If our calculated value is smaller than -128,
        // clamp it to -128.
        // ----------------------------------------------------

        else if (shifted_value < -128) begin


            output_value = -8'sd128;


        end


        // ----------------------------------------------------
        // Otherwise the result already fits into INT8.
        // ----------------------------------------------------

        else begin


            output_value =
                shifted_value[7:0];


        end


    end


endmodule