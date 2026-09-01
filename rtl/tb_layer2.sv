// ============================================================
// TinyNPU - Layer 2 Testbench
// ============================================================
//
// Tests:
//
//     14 INT8 inputs
//             ↓
//          7 neurons
//             ↓
//      7 INT8 outputs
//
// Test values:
//
//     Every input  = 100
//     Every weight = 1
//     Every bias   = 0
//
// Therefore:
//
//     MAC = 14 × 100
//         = 1400
//
// Requantization:
//
//     (1400 × 1223) >>> 20
//     ≈ 1
//
// ReLU:
//
//     1 → 1
//
// Expected:
//
//     All seven outputs = 1
//
// ============================================================


module tb_layer2;


    // ========================================================
    // PARAMETERS
    // ========================================================

    // Number of inputs entering Layer 2.
    localparam integer INPUTS = 14;


    // Number of neurons in Layer 2.
    localparam integer NEURONS = 7;


    // ========================================================
    // INPUT BUS
    // ========================================================

    // 14 INT8 values.
    //
    // 14 × 8 = 112 bits.
    logic signed [(INPUTS*8)-1:0] inputs_bus;


    // ========================================================
    // WEIGHT BUS
    // ========================================================

    // 7 × 14 INT8 weights.
    //
    // 7 × 14 × 8 = 784 bits.
    logic signed [(INPUTS*NEURONS*8)-1:0] weights_bus;


    // ========================================================
    // BIAS BUS
    // ========================================================

    // 7 INT32 biases.
    //
    // 7 × 32 = 224 bits.
    logic signed [(NEURONS*32)-1:0] biases_bus;


    // ========================================================
    // OUTPUT BUS
    // ========================================================

    // 7 INT8 outputs.
    //
    // 7 × 8 = 56 bits.
    logic signed [(NEURONS*8)-1:0] outputs_bus;


    // ========================================================
    // LOOP VARIABLES
    // ========================================================

    // Loop counter.
    integer i;


    // Number of failed tests.
    integer failed;


    // ========================================================
    // DEVICE UNDER TEST
    // ========================================================

    layer2 #(

        // 14 inputs.
        .INPUTS(INPUTS),

        // 7 neurons.
        .NEURONS(NEURONS),

        // Test requantization multiplier.
        .MULTIPLIER(1223),

        // Test requantization shift.
        .SHIFT(20)

    ) dut (

        // Connect input bus.
        .inputs_bus(inputs_bus),

        // Connect weight bus.
        .weights_bus(weights_bus),

        // Connect bias bus.
        .biases_bus(biases_bus),

        // Connect output bus.
        .outputs_bus(outputs_bus)

    );


    // ========================================================
    // TEST PROCEDURE
    // ========================================================

    initial begin


        // ----------------------------------------------------
        // Start with zero failures.
        // ----------------------------------------------------

        failed = 0;


        // ----------------------------------------------------
        // Print test header.
        // ----------------------------------------------------

        $display("====================================");

        $display("TinyNPU Layer 2 Test");

        $display("====================================");


        // ====================================================
        // INITIALIZE INPUT BUS
        // ====================================================

        // Clear the complete bus.
        inputs_bus = '0;


        // ----------------------------------------------------
        // Write 100 into each of the 14 INT8 positions.
        // ----------------------------------------------------

        for (
            i = 0;
            i < INPUTS;
            i = i + 1
        ) begin


            // Each input occupies eight bits.
            inputs_bus[i*8 +: 8] = 8'sd100;


        end


        // ====================================================
        // INITIALIZE WEIGHT BUS
        // ====================================================

        // Clear all 784 bits first.
        weights_bus = '0;


        // ----------------------------------------------------
        // Put value 1 into every weight.
        // ----------------------------------------------------

        for (
            i = 0;
            i < INPUTS*NEURONS;
            i = i + 1
        ) begin


            // Every weight occupies eight bits.
            weights_bus[i*8 +: 8] = 8'sd1;


        end


        // ====================================================
        // INITIALIZE BIASES
        // ====================================================

        // Set every one of the seven biases to zero.
        biases_bus = '0;


        // ====================================================
        // WAIT FOR HARDWARE
        // ====================================================

        // Layer 2 is combinational.
        //
        // Wait for signals to propagate through:
        //
        //     7 neurons
        //       ↓
        //      MAC
        //       ↓
        //   requantization
        //       ↓
        //      ReLU
        //
        #1;


        // ====================================================
        // DISPLAY OUTPUTS
        // ====================================================

        $display("");

        $display("Layer 2 Outputs:");

        $display("------------------------------------");


        // Display all seven outputs.
        for (
            i = 0;
            i < NEURONS;
            i = i + 1
        ) begin


            $display(
                "Neuron %0d output = %0d",
                i,
                $signed(outputs_bus[i*8 +: 8])
            );


        end


        // ====================================================
        // VERIFY OUTPUTS
        // ====================================================

        $display("");

        $display("Verification:");

        $display("------------------------------------");


        // Check all seven neurons.
        for (
            i = 0;
            i < NEURONS;
            i = i + 1
        ) begin


            // ------------------------------------------------
            // Detect X or Z.
            // ------------------------------------------------

            if (
                ^outputs_bus[i*8 +: 8]
                ===
                1'bx
            ) begin


                // Count failure.
                failed = failed + 1;


                // Report unknown output.
                $display(
                    "Neuron %0d FAILED: output is X/Z",
                    i
                );


            end


            // ------------------------------------------------
            // Check numerical value.
            // ------------------------------------------------

            else if (
                $signed(outputs_bus[i*8 +: 8])
                !==
                8'sd1
            ) begin


                // Count failure.
                failed = failed + 1;


                // Report incorrect value.
                $display(
                    "Neuron %0d FAILED: expected 1, got %0d",
                    i,
                    $signed(outputs_bus[i*8 +: 8])
                );


            end


            // ------------------------------------------------
            // Correct result.
            // ------------------------------------------------

            else begin


                $display(
                    "Neuron %0d PASSED",
                    i
                );


            end


        end


        // ====================================================
        // FINAL RESULT
        // ====================================================

        $display("");

        $display("====================================");


        // Check whether every neuron passed.
        if (failed == 0) begin


            $display("LAYER 2 TEST PASSED");


            $display(
                "%0d/%0d NEURONS PASSED",
                NEURONS,
                NEURONS
            );


        end
        else begin


            $display("LAYER 2 TEST FAILED");


            $display(
                "Failed neurons = %0d",
                failed
            );


        end


        $display("====================================");


        // ----------------------------------------------------
        // Stop simulation.
        // ----------------------------------------------------

        $finish;


    end


endmodule