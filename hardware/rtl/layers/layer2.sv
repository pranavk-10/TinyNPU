// ============================================================
// TinyNPU - Layer 2
// ============================================================
//
// Neural-network architecture:
//
//     14 INT8 inputs
//             |
//             v
//       +-------------+
//       |  7 neurons  |
//       +-------------+
//             |
//             v
//      7 INT8 outputs
//
// Weight count:
//
//     14 inputs × 7 neurons
//
//     = 98 weights
//
// Packed-bus architecture:
//
//     inputs_bus
//         14 × 8
//         = 112 bits
//
//     weights_bus
//         7 × 14 × 8
//         = 784 bits
//
//     biases_bus
//         7 × 32
//         = 224 bits
//
//     outputs_bus
//         7 × 8
//         = 56 bits
//
// ============================================================


module layer2 #(

    // --------------------------------------------------------
    // Number of inputs entering Layer 2.
    //
    // These are the 14 outputs from Layer 1.
    // --------------------------------------------------------

    parameter integer INPUTS = 14,


    // --------------------------------------------------------
    // Number of neurons in Layer 2.
    // --------------------------------------------------------

    parameter integer NEURONS = 7,


    // --------------------------------------------------------
    // Requantization multiplier.
    //
    // Temporary test value.
    //
    // We will later replace this with the exact Layer 2
    // fixed-point multiplier generated from Python.
    // --------------------------------------------------------

    parameter integer MULTIPLIER = 1223,


    // --------------------------------------------------------
    // Requantization shift.
    // --------------------------------------------------------

    parameter integer SHIFT = 20

)(

    // ========================================================
    // INPUT BUS
    // ========================================================

    // --------------------------------------------------------
    // 14 INT8 inputs packed into one bus.
    //
    // 14 × 8 = 112 bits.
    // --------------------------------------------------------

    input logic signed [(INPUTS*8)-1:0] inputs_bus,


    // ========================================================
    // WEIGHT BUS
    // ========================================================

    // --------------------------------------------------------
    // 7 neurons × 14 weights × 8 bits.
    //
    // 7 × 14 × 8 = 784 bits.
    // --------------------------------------------------------

    input logic signed [(INPUTS*NEURONS*8)-1:0] weights_bus,


    // ========================================================
    // BIAS BUS
    // ========================================================

    // --------------------------------------------------------
    // 7 INT32 biases.
    //
    // 7 × 32 = 224 bits.
    // --------------------------------------------------------

    input logic signed [(NEURONS*32)-1:0] biases_bus,


    // ========================================================
    // OUTPUT BUS
    // ========================================================

    // --------------------------------------------------------
    // 7 INT8 outputs.
    //
    // 7 × 8 = 56 bits.
    // --------------------------------------------------------

    output logic signed [(NEURONS*8)-1:0] outputs_bus

);


    // ========================================================
    // GENERATE 7 NEURONS
    // ========================================================

    // --------------------------------------------------------
    // Generate variable used to create the seven physical
    // neuron instances.
    // --------------------------------------------------------

    genvar n;


    // --------------------------------------------------------
    // Start hardware generation.
    // --------------------------------------------------------

    generate


        // ----------------------------------------------------
        // Create neurons 0 through 6.
        // ----------------------------------------------------

        for (
            n = 0;
            n < NEURONS;
            n = n + 1
        ) begin : neuron_instances


            // =================================================
            // LOCAL WEIGHT BUS
            // =================================================

            // -------------------------------------------------
            // Every neuron requires 14 INT8 weights.
            //
            // 14 × 8 = 112 bits.
            // -------------------------------------------------

            wire signed [(INPUTS*8)-1:0] neuron_weights;


            // -------------------------------------------------
            // Select the appropriate 112-bit section from the
            // complete 784-bit weight bus.
            //
            // Neuron 0:
            //
            //     weights[0:13]
            //
            // Neuron 1:
            //
            //     weights[14:27]
            //
            // ...
            //
            // Neuron 6:
            //
            //     weights[84:97]
            // -------------------------------------------------

            assign neuron_weights =
                weights_bus[
                    n*(INPUTS*8) +: (INPUTS*8)
                ];


            // =================================================
            // LOCAL BIAS
            // =================================================

            // -------------------------------------------------
            // Each neuron needs one 32-bit bias.
            // -------------------------------------------------

            wire signed [31:0] neuron_bias;


            // -------------------------------------------------
            // Select this neuron's bias from the packed bus.
            //
            // Neuron 0 → bits 31:0
            // Neuron 1 → bits 63:32
            // ...
            // Neuron 6 → bits 223:192
            // -------------------------------------------------

            assign neuron_bias =
                biases_bus[
                    n*32 +: 32
                ];


            // =================================================
            // NEURON INSTANCE
            // =================================================

            // -------------------------------------------------
            // Instantiate the neuron module.
            // -------------------------------------------------

            neuron #(

                // Layer 2 neurons have 14 inputs.
                .N(INPUTS),

                // Pass requantization multiplier.
                .MULTIPLIER(MULTIPLIER),

                // Pass requantization shift.
                .SHIFT(SHIFT)

            ) neuron_unit (

                // ------------------------------------------------
                // All seven neurons receive the same 14 inputs.
                // ------------------------------------------------

                .inputs_bus(inputs_bus),


                // ------------------------------------------------
                // This neuron receives its own 14 weights.
                // ------------------------------------------------

                .weights_bus(neuron_weights),


                // ------------------------------------------------
                // This neuron receives its own bias.
                // ------------------------------------------------

                .bias(neuron_bias),


                // ------------------------------------------------
                // Connect this neuron's output to its 8-bit
                // section of the Layer 2 output bus.
                // ------------------------------------------------

                .output_value(
                    outputs_bus[n*8 +: 8]
                )

            );


        end


    endgenerate


endmodule