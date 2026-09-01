// ============================================================
// TinyNPU - Layer 1
// ============================================================
//
// Architecture:
//
//     21 INT8 inputs
//            ↓
//     ┌──────────────┐
//     │ 14 neurons   │
//     └──────────────┘
//            ↓
//     14 INT8 outputs
//
// This implementation uses PACKED BUSES.
//
// This avoids Icarus issues with unpacked array ports.
//
// ============================================================


module layer1 #(

    // Number of inputs.
    parameter integer INPUTS = 21,

    // Number of neurons.
    parameter integer NEURONS = 14,

    // Requantization multiplier.
    parameter integer MULTIPLIER = 1223,

    // Requantization shift.
    parameter integer SHIFT = 20

)(

    // --------------------------------------------------------
    // Packed input bus.
    //
    // 21 × 8 = 168 bits.
    // --------------------------------------------------------

    input logic signed [(INPUTS*8)-1:0] inputs_bus,


    // --------------------------------------------------------
    // Packed weight bus.
    //
    // 14 × 21 × 8
    //
    // = 2352 bits.
    // --------------------------------------------------------

    input logic signed [(INPUTS*NEURONS*8)-1:0] weights_bus,


    // --------------------------------------------------------
    // 14 INT32 biases.
    //
    // 14 × 32 = 448 bits.
    // --------------------------------------------------------

    input logic signed [(NEURONS*32)-1:0] biases_bus,


    // --------------------------------------------------------
    // 14 INT8 outputs.
    //
    // 14 × 8 = 112 bits.
    // --------------------------------------------------------

    output logic signed [(NEURONS*8)-1:0] outputs_bus

);


    // ========================================================
    // GENERATE 14 NEURONS
    // ========================================================

    genvar n;

    generate

        for (
            n = 0;
            n < NEURONS;
            n = n + 1
        ) begin : neuron_instances


            // ------------------------------------------------
            // Each neuron gets one local 21 × 8-bit weight bus.
            // ------------------------------------------------

            wire signed [(INPUTS*8)-1:0] neuron_weights;


            // ------------------------------------------------
            // Connect the appropriate section of the large
            // weight bus to this neuron.
            //
            // Each neuron requires:
            //
            //     INPUTS × 8
            //
            // bits.
            // ------------------------------------------------

            assign neuron_weights =
                weights_bus[
                    n*(INPUTS*8) +: (INPUTS*8)
                ];


            // ------------------------------------------------
            // Extract this neuron's 32-bit bias.
            // ------------------------------------------------

            wire signed [31:0] neuron_bias;


            assign neuron_bias =
                biases_bus[
                    n*32 +: 32
                ];


            // ------------------------------------------------
            // Instantiate the neuron.
            // ------------------------------------------------

            neuron #(

                // 21 inputs.
                .N(INPUTS),

                // Requantization multiplier.
                .MULTIPLIER(MULTIPLIER),

                // Requantization shift.
                .SHIFT(SHIFT)

            ) neuron_unit (

                // Connect the complete 21-input bus.
                .inputs_bus(inputs_bus),

                // Connect this neuron's 21 weights.
                .weights_bus(neuron_weights),

                // Connect this neuron's bias.
                .bias(neuron_bias),

                // Connect this neuron's output to its
                // corresponding 8-bit section of outputs_bus.
                .output_value(
                    outputs_bus[n*8 +: 8]
                )

            );


        end

    endgenerate


endmodule