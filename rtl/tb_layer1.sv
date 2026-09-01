// ============================================================
// TinyNPU - Layer 1 Testbench
// ============================================================
//
// Test:
//
//     21 inputs → 14 neurons
//
// Every input:
//
//     100
//
// Every weight:
//
//     1
//
// Every bias:
//
//     0
//
// Therefore:
//
//     MAC = 21 × 100
//         = 2100
//
// Requantization:
//
//     2100 × 1223 / 2^20
//     = 2
//
// ReLU:
//
//     2 → 2
//
// Expected:
//
//     All 14 outputs = 2
//
// ============================================================


module tb_layer1;


    // ========================================================
    // PARAMETERS
    // ========================================================

    localparam integer INPUTS = 21;

    localparam integer NEURONS = 14;


    // ========================================================
    // BUSES
    // ========================================================

    // --------------------------------------------------------
    // 21 × 8 = 168-bit input bus.
    // --------------------------------------------------------

    logic signed [(INPUTS*8)-1:0] inputs_bus;


    // --------------------------------------------------------
    // 14 × 21 × 8 = 2352-bit weight bus.
    // --------------------------------------------------------

    logic signed [(INPUTS*NEURONS*8)-1:0] weights_bus;


    // --------------------------------------------------------
    // 14 × 32 = 448-bit bias bus.
    // --------------------------------------------------------

    logic signed [(NEURONS*32)-1:0] biases_bus;


    // --------------------------------------------------------
    // 14 × 8 = 112-bit output bus.
    // --------------------------------------------------------

    logic signed [(NEURONS*8)-1:0] outputs_bus;


    // ========================================================
    // LOOP VARIABLES
    // ========================================================

    integer i;

    integer failed;


    // ========================================================
    // DEVICE UNDER TEST
    // ========================================================

    layer1 #(

        .INPUTS(INPUTS),

        .NEURONS(NEURONS),

        .MULTIPLIER(1223),

        .SHIFT(20)

    ) dut (

        .inputs_bus(inputs_bus),

        .weights_bus(weights_bus),

        .biases_bus(biases_bus),

        .outputs_bus(outputs_bus)

    );


    // ========================================================
    // TEST
    // ========================================================

    initial begin


        // ----------------------------------------------------
        // Start with zero failures.
        // ----------------------------------------------------

        failed = 0;


        $display("====================================");

        $display("TinyNPU Layer 1 Test");

        $display("====================================");


        // ====================================================
        // INITIALIZE INPUTS
        // ====================================================

        // Clear the entire input bus first.
        inputs_bus = '0;


        // Put 100 into every INT8 input slot.
        for (
            i = 0;
            i < INPUTS;
            i = i + 1
        ) begin

            inputs_bus[i*8 +: 8] = 8'sd100;

        end


        // ====================================================
        // INITIALIZE WEIGHTS
        // ====================================================

        // Clear entire weight bus.
        weights_bus = '0;


        // Put 1 into every INT8 weight slot.
        for (
            i = 0;
            i < INPUTS*NEURONS;
            i = i + 1
        ) begin

            weights_bus[i*8 +: 8] = 8'sd1;

        end


        // ====================================================
        // INITIALIZE BIASES
        // ====================================================

        // Clear all 14 biases.
        biases_bus = '0;


        // ====================================================
        // WAIT FOR COMBINATIONAL LOGIC
        // ====================================================

        #1;


        // ====================================================
        // DISPLAY OUTPUTS
        // ====================================================

        $display("");

        $display("Layer 1 Outputs:");

        $display("------------------------------------");


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


        for (
            i = 0;
            i < NEURONS;
            i = i + 1
        ) begin


            // ------------------------------------------------
            // Detect unknown X/Z output.
            // ------------------------------------------------

            if (
                ^outputs_bus[i*8 +: 8]
                ===
                1'bx
            ) begin


                failed = failed + 1;


                $display(
                    "Neuron %0d FAILED: output is X/Z",
                    i
                );


            end


            // ------------------------------------------------
            // Check expected value.
            // ------------------------------------------------

            else if (
                $signed(outputs_bus[i*8 +: 8])
                !==
                8'sd2
            ) begin


                failed = failed + 1;


                $display(
                    "Neuron %0d FAILED: expected 2, got %0d",
                    i,
                    $signed(outputs_bus[i*8 +: 8])
                );


            end


            // ------------------------------------------------
            // Correct.
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


        if (failed == 0) begin


            $display("LAYER 1 TEST PASSED");

            $display(
                "%0d/%0d NEURONS PASSED",
                NEURONS,
                NEURONS
            );


        end
        else begin


            $display("LAYER 1 TEST FAILED");

            $display(
                "Failed neurons = %0d",
                failed
            );


        end


        $display("====================================");


        // ----------------------------------------------------
        // End simulation.
        // ----------------------------------------------------

        $finish;


    end


endmodule