// ============================================================
// TinyNPU - Layer 3
// ============================================================
//
// Final neural-network layer:
//
//     7 INT8 inputs
//            |
//            v
//        1 neuron
//            |
//            v
//       1 INT8 output
//
// Weight count:
//
//     7 × 1 = 7 weights
//
// Packed buses:
//
//     inputs_bus  = 7 × 8  = 56 bits
//     weights_bus = 7 × 8  = 56 bits
//     bias        = 32 bits
//     output      = 8 bits
//
// ============================================================


module layer3 #(

    // Number of inputs entering Layer 3.
    parameter integer INPUTS = 7,

    // Layer 3 contains one neuron.
    parameter integer NEURONS = 1,

    // Requantization multiplier.
    parameter integer MULTIPLIER = 1223,

    // Requantization shift.
    parameter integer SHIFT = 20

)(

    // --------------------------------------------------------
    // Seven INT8 inputs packed into one bus.
    // --------------------------------------------------------

    input logic signed [(INPUTS*8)-1:0] inputs_bus,


    // --------------------------------------------------------
    // Seven INT8 weights packed into one bus.
    //
    // 7 × 8 = 56 bits.
    // --------------------------------------------------------

    input logic signed [(INPUTS*NEURONS*8)-1:0] weights_bus,


    // --------------------------------------------------------
    // One INT32 bias.
    //
    // Because Layer 3 has only one neuron, we don't need a
    // multi-bias array here.
    // --------------------------------------------------------

    input logic signed [31:0] bias,


    // --------------------------------------------------------
    // One INT8 output.
    // --------------------------------------------------------

    output logic signed [7:0] output_value

);


    // ========================================================
    // NEURON
    // ========================================================

    // --------------------------------------------------------
    // Instantiate the final neuron.
    // --------------------------------------------------------

    neuron #(

        // Layer 3 has seven inputs.
        .N(INPUTS),

        // Requantization multiplier.
        .MULTIPLIER(MULTIPLIER),

        // Requantization shift.
        .SHIFT(SHIFT)

    ) neuron_unit (

        // Connect seven inputs.
        .inputs_bus(inputs_bus),

        // Connect seven weights.
        .weights_bus(weights_bus),

        // Connect the final-layer bias.
        .bias(bias),

        // Connect final output.
        .output_value(output_value)

    );


endmodule