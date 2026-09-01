// ============================================================
// TinyNPU - Top Level Neural Network
// ============================================================
//
// Complete neural network:
//
//     21 INT8 inputs
//            |
//            v
//       Layer 1
//       21 -> 14
//            |
//            v
//       Layer 2
//       14 -> 7
//            |
//            v
//       Layer 3
//        7 -> 1
//            |
//            v
//       1 INT8 output
//
// ============================================================
//
// IMPORTANT:
//
// This version uses packed buses throughout the hierarchy.
//
// Layer 1:
//     Input  = 21 × 8  = 168 bits
//     Output = 14 × 8  = 112 bits
//
// Layer 2:
//     Input  = 14 × 8  = 112 bits
//     Output = 7 × 8   = 56 bits
//
// Layer 3:
//     Input  = 7 × 8   = 56 bits
//     Output = 1 × 8   = 8 bits
//
// ============================================================


module tinynpu #(

    // --------------------------------------------------------
    // Input layer size.
    // --------------------------------------------------------

    parameter integer INPUTS = 21,

    // --------------------------------------------------------
    // Layer 1 neuron count.
    // --------------------------------------------------------

    parameter integer LAYER1_NEURONS = 14,

    // --------------------------------------------------------
    // Layer 2 neuron count.
    // --------------------------------------------------------

    parameter integer LAYER2_NEURONS = 7,

    // --------------------------------------------------------
    // Layer 3 neuron count.
    // --------------------------------------------------------

    parameter integer LAYER3_NEURONS = 1,


    // --------------------------------------------------------
    // Layer 1 requantization parameters.
    // --------------------------------------------------------

    parameter integer L1_MULTIPLIER = 1223,

    parameter integer L1_SHIFT = 20,


    // --------------------------------------------------------
    // Layer 2 requantization parameters.
    // --------------------------------------------------------

    parameter integer L2_MULTIPLIER = 1223,

    parameter integer L2_SHIFT = 20,


    // --------------------------------------------------------
    // Layer 3 requantization parameters.
    // --------------------------------------------------------

    parameter integer L3_MULTIPLIER = 1223,

    parameter integer L3_SHIFT = 20

)(

    // ========================================================
    // TOP-LEVEL INPUT
    // ========================================================

    // 21 INT8 input values.
    //
    // 21 × 8 = 168 bits.
    input logic signed [(INPUTS*8)-1:0] inputs_bus,


    // ========================================================
    // LAYER 1 PARAMETERS
    // ========================================================

    // 294 INT8 weights.
    //
    // 21 × 14 × 8 = 2352 bits.
    input logic signed
        [(INPUTS*LAYER1_NEURONS*8)-1:0] weights1_bus,


    // 14 INT32 biases.
    //
    // 14 × 32 = 448 bits.
    input logic signed
        [(LAYER1_NEURONS*32)-1:0] biases1_bus,


    // ========================================================
    // LAYER 2 PARAMETERS
    // ========================================================

    // 98 INT8 weights.
    //
    // 14 × 7 × 8 = 784 bits.
    input logic signed
        [(LAYER1_NEURONS*LAYER2_NEURONS*8)-1:0] weights2_bus,


    // 7 INT32 biases.
    //
    // 7 × 32 = 224 bits.
    input logic signed
        [(LAYER2_NEURONS*32)-1:0] biases2_bus,


    // ========================================================
    // LAYER 3 PARAMETERS
    // ========================================================

    // 7 INT8 weights.
    //
    // 7 × 1 × 8 = 56 bits.
    input logic signed
        [(LAYER2_NEURONS*LAYER3_NEURONS*8)-1:0] weights3_bus,


    // One INT32 bias.
    input logic signed [31:0] bias3,


    // ========================================================
    // FINAL OUTPUT
    // ========================================================

    // Final INT8 neural-network output.
    output logic signed [7:0] output_value

);


    // ========================================================
    // INTERNAL LAYER 1 OUTPUT
    // ========================================================

    // --------------------------------------------------------
    // Layer 1 produces 14 INT8 activations.
    //
    // 14 × 8 = 112 bits.
    // --------------------------------------------------------

    logic signed
        [(LAYER1_NEURONS*8)-1:0] layer1_output;


    // ========================================================
    // INTERNAL LAYER 2 OUTPUT
    // ========================================================

    // --------------------------------------------------------
    // Layer 2 produces 7 INT8 activations.
    //
    // 7 × 8 = 56 bits.
    // --------------------------------------------------------

    logic signed
        [(LAYER2_NEURONS*8)-1:0] layer2_output;


    // ========================================================
    // LAYER 1
    // ========================================================

    // --------------------------------------------------------
    // Instantiate Layer 1.
    //
    // 21 → 14
    // --------------------------------------------------------

    layer1 #(

        // Layer 1 input count.
        .INPUTS(INPUTS),

        // Layer 1 neuron count.
        .NEURONS(LAYER1_NEURONS),

        // Layer 1 multiplier.
        .MULTIPLIER(L1_MULTIPLIER),

        // Layer 1 shift.
        .SHIFT(L1_SHIFT)

    ) layer1_unit (

        // Top-level inputs enter Layer 1.
        .inputs_bus(inputs_bus),

        // Layer 1 weights.
        .weights_bus(weights1_bus),

        // Layer 1 biases.
        .biases_bus(biases1_bus),

        // Layer 1 produces 14 activations.
        .outputs_bus(layer1_output)

    );


    // ========================================================
    // LAYER 2
    // ========================================================

    // --------------------------------------------------------
    // Instantiate Layer 2.
    //
    // 14 → 7
    //
    // The output of Layer 1 becomes the input of Layer 2.
    // --------------------------------------------------------

    layer2 #(

        // Layer 2 receives 14 values.
        .INPUTS(LAYER1_NEURONS),

        // Layer 2 has 7 neurons.
        .NEURONS(LAYER2_NEURONS),

        // Layer 2 multiplier.
        .MULTIPLIER(L2_MULTIPLIER),

        // Layer 2 shift.
        .SHIFT(L2_SHIFT)

    ) layer2_unit (

        // Layer 1 output feeds Layer 2 input.
        .inputs_bus(layer1_output),

        // Layer 2 weights.
        .weights_bus(weights2_bus),

        // Layer 2 biases.
        .biases_bus(biases2_bus),

        // Layer 2 produces 7 activations.
        .outputs_bus(layer2_output)

    );


    // ========================================================
    // LAYER 3
    // ========================================================

    // --------------------------------------------------------
    // Instantiate Layer 3.
    //
    // 7 → 1
    //
    // Layer 2 output becomes Layer 3 input.
    // --------------------------------------------------------

    layer3 #(

        // Layer 3 receives 7 values.
        .INPUTS(LAYER2_NEURONS),

        // Layer 3 has one neuron.
        .NEURONS(LAYER3_NEURONS),

        // Layer 3 multiplier.
        .MULTIPLIER(L3_MULTIPLIER),

        // Layer 3 shift.
        .SHIFT(L3_SHIFT)

    ) layer3_unit (

        // Layer 2 output feeds Layer 3.
        .inputs_bus(layer2_output),

        // Layer 3 weights.
        .weights_bus(weights3_bus),

        // Layer 3 bias.
        .bias(bias3),

        // Final network output.
        .output_value(output_value)

    );


endmodule