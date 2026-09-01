// ============================================================
// TinyNPU - Complete Network Testbench
// ============================================================
//
// Tests the entire network:
//
//     21 → 14 → 7 → 1
//
// Every input:
//     1
//
// Every weight:
//     1
//
// Every bias:
//     0
//
// Therefore:
//
// Layer 1:
//     21 × 1 = 21
//
// Layer 2:
//     14 × 1 = 14
//
// Layer 3:
//     7 × 1 = 7
//
// With the current test requantization:
//
//     1223 / 2^20
//
// all three values requantize to 0.
//
// Expected final output:
//
//     0
//
// ============================================================


module tb_tinynpu;


    // ========================================================
    // NETWORK PARAMETERS
    // ========================================================

    localparam integer INPUTS = 21;

    localparam integer LAYER1_NEURONS = 14;

    localparam integer LAYER2_NEURONS = 7;

    localparam integer LAYER3_NEURONS = 1;


    // ========================================================
    // TOP-LEVEL INPUT BUS
    // ========================================================

    // 21 × 8 = 168 bits.
    logic signed [(INPUTS*8)-1:0] inputs_bus;


    // ========================================================
    // LAYER 1 WEIGHTS
    // ========================================================

    // 21 × 14 × 8 = 2352 bits.
    logic signed
        [(INPUTS*LAYER1_NEURONS*8)-1:0] weights1_bus;


    // ========================================================
    // LAYER 1 BIASES
    // ========================================================

    // 14 × 32 = 448 bits.
    logic signed
        [(LAYER1_NEURONS*32)-1:0] biases1_bus;


    // ========================================================
    // LAYER 2 WEIGHTS
    // ========================================================

    // 14 × 7 × 8 = 784 bits.
    logic signed
        [(LAYER1_NEURONS*LAYER2_NEURONS*8)-1:0] weights2_bus;


    // ========================================================
    // LAYER 2 BIASES
    // ========================================================

    // 7 × 32 = 224 bits.
    logic signed
        [(LAYER2_NEURONS*32)-1:0] biases2_bus;


    // ========================================================
    // LAYER 3 WEIGHTS
    // ========================================================

    // 7 × 1 × 8 = 56 bits.
    logic signed
        [(LAYER2_NEURONS*LAYER3_NEURONS*8)-1:0] weights3_bus;


    // ========================================================
    // LAYER 3 BIAS
    // ========================================================

    // One INT32 bias.
    logic signed [31:0] bias3;


    // ========================================================
    // FINAL OUTPUT
    // ========================================================

    // Final INT8 prediction value.
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

    tinynpu dut (

        // ----------------------------------------------------
        // Network input.
        // ----------------------------------------------------

        .inputs_bus(inputs_bus),


        // ----------------------------------------------------
        // Layer 1 parameters.
        // ----------------------------------------------------

        .weights1_bus(weights1_bus),

        .biases1_bus(biases1_bus),


        // ----------------------------------------------------
        // Layer 2 parameters.
        // ----------------------------------------------------

        .weights2_bus(weights2_bus),

        .biases2_bus(biases2_bus),


        // ----------------------------------------------------
        // Layer 3 parameters.
        // ----------------------------------------------------

        .weights3_bus(weights3_bus),

        .bias3(bias3),


        // ----------------------------------------------------
        // Final output.
        // ----------------------------------------------------

        .output_value(output_value)

    );


    // ========================================================
    // TEST
    // ========================================================

    initial begin


        // ----------------------------------------------------
        // Start with zero failures.
        // ----------------------------------------------------

        failed = 0;


        // ----------------------------------------------------
        // Print header.
        // ----------------------------------------------------

        $display("====================================");

        $display("TinyNPU Complete Network Test");

        $display("====================================");


        $display("");

        $display("Architecture:");

        $display("21 -> 14 -> 7 -> 1");

        $display("");


        // ====================================================
        // INITIALIZE INPUTS
        // ====================================================

        // Clear input bus.
        inputs_bus = '0;


        // ----------------------------------------------------
        // Set all 21 inputs to 1.
        // ----------------------------------------------------

        for (
            i = 0;
            i < INPUTS;
            i = i + 1
        ) begin

            inputs_bus[i*8 +: 8] = 8'sd1;

        end


        // ====================================================
        // INITIALIZE LAYER 1 WEIGHTS
        // ====================================================

        // Clear Layer 1 weight bus.
        weights1_bus = '0;


        // ----------------------------------------------------
        // Set all 294 weights to 1.
        // ----------------------------------------------------

        for (
            i = 0;
            i < INPUTS*LAYER1_NEURONS;
            i = i + 1
        ) begin

            weights1_bus[i*8 +: 8] = 8'sd1;

        end


        // ====================================================
        // INITIALIZE LAYER 1 BIASES
        // ====================================================

        // Every Layer 1 bias = 0.
        biases1_bus = '0;


        // ====================================================
        // INITIALIZE LAYER 2 WEIGHTS
        // ====================================================

        // Clear Layer 2 weights.
        weights2_bus = '0;


        // ----------------------------------------------------
        // Set all 98 Layer 2 weights to 1.
        // ----------------------------------------------------

        for (
            i = 0;
            i < LAYER1_NEURONS*LAYER2_NEURONS;
            i = i + 1
        ) begin

            weights2_bus[i*8 +: 8] = 8'sd1;

        end


        // ====================================================
        // INITIALIZE LAYER 2 BIASES
        // ====================================================

        // Every Layer 2 bias = 0.
        biases2_bus = '0;


        // ====================================================
        // INITIALIZE LAYER 3 WEIGHTS
        // ====================================================

        // Clear Layer 3 weights.
        weights3_bus = '0;


        // ----------------------------------------------------
        // Set all seven Layer 3 weights to 1.
        // ----------------------------------------------------

        for (
            i = 0;
            i < LAYER2_NEURONS;
            i = i + 1
        ) begin

            weights3_bus[i*8 +: 8] = 8'sd1;

        end


        // ====================================================
        // INITIALIZE LAYER 3 BIAS
        // ====================================================

        bias3 = 32'sd0;


        // ====================================================
        // WAIT FOR COMBINATIONAL NETWORK
        // ====================================================

        // Give all three layers time to propagate.
        #1;


        // ====================================================
        // DISPLAY INTERNAL LAYER OUTPUTS
        // ====================================================

        $display("Layer 1 Output:");

        // Display the first Layer 1 neuron.
        $display(
            "  Neuron 0 = %0d",
            $signed(
                dut.layer1_output[0*8 +: 8]
            )
        );


        $display("");

        $display("Layer 2 Output:");

        // Display the first Layer 2 neuron.
        $display(
            "  Neuron 0 = %0d",
            $signed(
                dut.layer2_output[0*8 +: 8]
            )
        );


        $display("");

        $display("Final Output:");

        $display(
            "  Output = %0d",
            $signed(output_value)
        );


        // ====================================================
        // CHECK FINAL OUTPUT FOR X/Z
        // ====================================================

        $display("");

        $display("Verification:");

        $display("------------------------------------");


        // ----------------------------------------------------
        // Detect unknown output.
        // ----------------------------------------------------

        if (^output_value === 1'bx) begin


            failed = failed + 1;


            $display(
                "TEST FAILED: final output is X/Z"
            );


        end


        // ====================================================
        // CHECK FINAL NUMERICAL RESULT
        // ====================================================

        else if (output_value !== 8'sd0) begin


            failed = failed + 1;


            $display(
                "TEST FAILED: expected 0, got %0d",
                $signed(output_value)
            );


        end


        // ====================================================
        // SUCCESS
        // ====================================================

        else begin


            $display(
                "FINAL OUTPUT CHECK PASSED"
            );


        end


        // ====================================================
        // FINAL RESULT
        // ====================================================

        $display("");

        $display("====================================");


        if (failed == 0) begin


            $display("TINY NPU INTEGRATION TEST PASSED");

            $display("21 -> 14 -> 7 -> 1");


        end
        else begin


            $display("TINY NPU INTEGRATION TEST FAILED");

            $display(
                "Failures = %0d",
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