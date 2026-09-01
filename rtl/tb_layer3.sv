// ============================================================
// TinyNPU - Layer 3 Testbench
// ============================================================
//
// Tests:
//
//     7 INT8 inputs
//           |
//           v
//        1 neuron
//           |
//           v
//       1 INT8 output
//
// Test values:
//
//     Every input  = 100
//     Every weight = 1
//     Bias         = 0
//
// MAC:
//
//     7 × 100 × 1
//     = 700
//
// Requantization:
//
//     (700 × 1223) >>> 20
//     = 0
//
// ReLU:
//
//     0 → 0
//
// Expected output:
//
//     0
//
// ============================================================


module tb_layer3;


    // ========================================================
    // PARAMETERS
    // ========================================================

    // Layer 3 has seven inputs.
    localparam integer INPUTS = 7;


    // Layer 3 has one neuron.
    localparam integer NEURONS = 1;


    // ========================================================
    // INPUT BUS
    // ========================================================

    // Seven INT8 values.
    //
    // 7 × 8 = 56 bits.
    logic signed [(INPUTS*8)-1:0] inputs_bus;


    // ========================================================
    // WEIGHT BUS
    // ========================================================

    // Seven INT8 weights.
    //
    // 7 × 8 = 56 bits.
    logic signed [(INPUTS*NEURONS*8)-1:0] weights_bus;


    // ========================================================
    // BIAS
    // ========================================================

    // One INT32 bias.
    logic signed [31:0] bias;


    // ========================================================
    // OUTPUT
    // ========================================================

    // One INT8 output.
    logic signed [7:0] output_value;


    // ========================================================
    // LOOP VARIABLE
    // ========================================================

    integer i;


    // ========================================================
    // FAILURE COUNTER
    // ========================================================

    integer failed;


    // ========================================================
    // DEVICE UNDER TEST
    // ========================================================

    layer3 #(

        // Seven inputs.
        .INPUTS(INPUTS),

        // One neuron.
        .NEURONS(NEURONS),

        // Test multiplier.
        .MULTIPLIER(1223),

        // Test shift.
        .SHIFT(20)

    ) dut (

        // Connect inputs.
        .inputs_bus(inputs_bus),

        // Connect weights.
        .weights_bus(weights_bus),

        // Connect bias.
        .bias(bias),

        // Connect output.
        .output_value(output_value)

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

        $display("TinyNPU Layer 3 Test");

        $display("====================================");


        // ====================================================
        // INITIALIZE INPUTS
        // ====================================================

        // Clear input bus.
        inputs_bus = '0;


        // Put 100 into each of the seven input positions.
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

        // Clear weight bus.
        weights_bus = '0;


        // Put 1 into every weight position.
        for (
            i = 0;
            i < INPUTS;
            i = i + 1
        ) begin

            weights_bus[i*8 +: 8] = 8'sd1;

        end


        // ====================================================
        // INITIALIZE BIAS
        // ====================================================

        // Set bias to zero.
        bias = 32'sd0;


        // ====================================================
        // WAIT FOR COMBINATIONAL LOGIC
        // ====================================================

        // Give the neuron time to calculate:
        //
        //     MAC
        //      ↓
        // Requantization
        //      ↓
        //     ReLU
        //
        #1;


        // ====================================================
        // DISPLAY RESULT
        // ====================================================

        $display("");

        $display("Layer 3 Output:");

        $display("------------------------------------");


        $display(
            "Output = %0d",
            $signed(output_value)
        );


        // ====================================================
        // VERIFY X/Z
        // ====================================================

        $display("");

        $display("Verification:");

        $display("------------------------------------");


        // ----------------------------------------------------
        // Detect unknown output.
        // ----------------------------------------------------

        if (^output_value === 1'bx) begin


            // Count failure.
            failed = failed + 1;


            // Report unknown state.
            $display(
                "TEST FAILED: output is X/Z"
            );


        end


        // ====================================================
        // VERIFY NUMERICAL VALUE
        // ====================================================

        else if (output_value !== 8'sd0) begin


            // Count failure.
            failed = failed + 1;


            // Report wrong result.
            $display(
                "TEST FAILED: expected 0, got %0d",
                $signed(output_value)
            );


        end


        // ====================================================
        // SUCCESS
        // ====================================================

        else begin


            // Expected result was obtained.
            $display(
                "TEST PASSED: output = 0"
            );


        end


        // ====================================================
        // FINAL RESULT
        // ====================================================

        $display("");

        $display("====================================");


        if (failed == 0) begin


            $display("LAYER 3 TEST PASSED");


        end
        else begin


            $display("LAYER 3 TEST FAILED");


        end


        $display("====================================");


        // ----------------------------------------------------
        // End simulation.
        // ----------------------------------------------------

        $finish;


    end


endmodule