// ============================================================
// TinyNPU - Real Model RTL Verification Testbench (Phase 7)
// ============================================================
//
// End-to-end golden verification against real trained and
// quantized neural network parameters:
//
//     1. Loads actual weights, biases, and test inputs using $readmemh
//     2. Packs parameters into flat top-level buses
//     3. Applies exact fixed-point requantization multipliers
//     4. Runs inference for all dataset samples
//     5. Compares layer-by-layer and final classification predictions
//        against Python reference model
// ============================================================

`timescale 1ns/1ps

module tb_tinynpu_real;

    // ========================================================
    // NETWORK PARAMETERS
    // ========================================================

    localparam integer INPUTS          = 21;
    localparam integer LAYER1_NEURONS  = 14;
    localparam integer LAYER2_NEURONS  = 7;
    localparam integer LAYER3_NEURONS  = 1;

    // ========================================================
    // REQUANTIZATION PARAMETERS (from Python calibration)
    // ========================================================

    `include "tinynpu_params.vh"

    localparam integer TOTAL_SAMPLES   = 2;

    // ========================================================
    // MEMORY ARRAYS FOR $readmemh
    // ========================================================

    logic signed [7:0]  mem_inputs   [0:(TOTAL_SAMPLES*INPUTS)-1];
    logic signed [7:0]  mem_weights1 [0:(INPUTS*LAYER1_NEURONS)-1];
    logic signed [31:0] mem_bias1    [0:LAYER1_NEURONS-1];
    logic signed [7:0]  mem_weights2 [0:(LAYER1_NEURONS*LAYER2_NEURONS)-1];
    logic signed [31:0] mem_bias2    [0:LAYER2_NEURONS-1];
    logic signed [7:0]  mem_weights3 [0:(LAYER2_NEURONS*LAYER3_NEURONS)-1];
    logic signed [31:0] mem_bias3    [0:LAYER3_NEURONS-1];

    logic signed [7:0]  mem_expected_classes [0:TOTAL_SAMPLES-1];
    logic signed [7:0]  mem_expected_l1      [0:(TOTAL_SAMPLES*LAYER1_NEURONS)-1];
    logic signed [7:0]  mem_expected_l2      [0:(TOTAL_SAMPLES*LAYER2_NEURONS)-1];

    // ========================================================
    // DUT SIGNALS
    // ========================================================

    logic signed [(INPUTS*8)-1:0]                         inputs_bus;
    logic signed [(INPUTS*LAYER1_NEURONS*8)-1:0]          weights1_bus;
    logic signed [(LAYER1_NEURONS*32)-1:0]                biases1_bus;
    logic signed [(LAYER1_NEURONS*LAYER2_NEURONS*8)-1:0]  weights2_bus;
    logic signed [(LAYER2_NEURONS*32)-1:0]                biases2_bus;
    logic signed [(LAYER2_NEURONS*LAYER3_NEURONS*8)-1:0]  weights3_bus;
    logic signed [31:0]                                   bias3;
    logic signed [7:0]                                    output_value;

    // ========================================================
    // INSTANTIATE TINYNPU DUT
    // ========================================================

    tinynpu #(
        .INPUTS(INPUTS),
        .LAYER1_NEURONS(LAYER1_NEURONS),
        .LAYER2_NEURONS(LAYER2_NEURONS),
        .LAYER3_NEURONS(LAYER3_NEURONS),
        .L1_MULTIPLIER(L1_MULTIPLIER),
        .L1_SHIFT(L1_SHIFT),
        .L2_MULTIPLIER(L2_MULTIPLIER),
        .L2_SHIFT(L2_SHIFT),
        .L3_MULTIPLIER(L3_MULTIPLIER),
        .L3_SHIFT(L3_SHIFT)
    ) dut (
        .inputs_bus(inputs_bus),
        .weights1_bus(weights1_bus),
        .biases1_bus(biases1_bus),
        .weights2_bus(weights2_bus),
        .biases2_bus(biases2_bus),
        .weights3_bus(weights3_bus),
        .bias3(bias3),
        .output_value(output_value)
    );

    // ========================================================
    // TESTBENCH STATE & STATS
    // ========================================================

    integer s, i, n;
    integer total_errors;
    integer classification_matches;
    logic signed [7:0] predicted_class;

    initial begin
        total_errors = 0;
        classification_matches = 0;

        $display("============================================================");
        $display("TinyNPU Phase 7: Real Model RTL Verification Testbench");
        $display("Architecture: 21 -> 14 -> 7 -> 1");
        $display("============================================================");

        // ----------------------------------------------------
        // 1. Load memory files into arrays
        // ----------------------------------------------------
        $display("\n[1] Loading .mem files into simulation memory...");
        $readmemh("data/mem/inputs.mem", mem_inputs);
        $readmemh("data/mem/weights1.mem", mem_weights1);
        $readmemh("data/mem/bias1.mem", mem_bias1);
        $readmemh("data/mem/weights2.mem", mem_weights2);
        $readmemh("data/mem/bias2.mem", mem_bias2);
        $readmemh("data/mem/weights3.mem", mem_weights3);
        $readmemh("data/mem/bias3.mem", mem_bias3);
        $readmemh("data/mem/expected_classes.mem", mem_expected_classes);
        $readmemh("data/mem/layer1_expected.mem", mem_expected_l1);
        $readmemh("data/mem/layer2_expected.mem", mem_expected_l2);
        $display("    All .mem files loaded successfully.");

        // ----------------------------------------------------
        // 2. Pack weights and biases into DUT buses
        // ----------------------------------------------------
        $display("\n[2] Packing parameters into hardware buses...");
        for (i = 0; i < (INPUTS * LAYER1_NEURONS); i = i + 1) begin
            weights1_bus[i*8 +: 8] = mem_weights1[i];
        end

        for (n = 0; n < LAYER1_NEURONS; n = n + 1) begin
            biases1_bus[n*32 +: 32] = mem_bias1[n];
        end

        for (i = 0; i < (LAYER1_NEURONS * LAYER2_NEURONS); i = i + 1) begin
            weights2_bus[i*8 +: 8] = mem_weights2[i];
        end

        for (n = 0; n < LAYER2_NEURONS; n = n + 1) begin
            biases2_bus[n*32 +: 32] = mem_bias2[n];
        end

        for (i = 0; i < (LAYER2_NEURONS * LAYER3_NEURONS); i = i + 1) begin
            weights3_bus[i*8 +: 8] = mem_weights3[i];
        end

        bias3 = mem_bias3[0];
        $display("    Hardware parameter buses initialized.");

        // ----------------------------------------------------
        // 3. Evaluate each sample from the dataset
        // ----------------------------------------------------
        $display("\n[3] Running Real-Model Inference Verification...");
        $display("------------------------------------------------------------");

        for (s = 0; s < TOTAL_SAMPLES; s = s + 1) begin
            // Load sample input into input bus
            for (i = 0; i < INPUTS; i = i + 1) begin
                inputs_bus[i*8 +: 8] = mem_inputs[s * INPUTS + i];
            end

            // Wait for combinational datapath to settle
            #10;

            // Prediction: Class 1 if output > 0, Class 0 if output == 0
            predicted_class = (output_value > 8'sd0) ? 8'sd1 : 8'sd0;

            $display("Sample %0d:", s + 1);
            $display("  Target Expected Class : %0d", mem_expected_classes[s]);
            $display("  RTL Output Activation : %0d", $signed(output_value));
            $display("  RTL Prediction        : %0d", predicted_class);

            // Layer 1 verification
            for (n = 0; n < LAYER1_NEURONS; n = n + 1) begin
                logic signed [7:0] l1_act;
                logic signed [7:0] l1_exp;
                l1_act = dut.layer1_output[n*8 +: 8];
                l1_exp = mem_expected_l1[s * LAYER1_NEURONS + n];
                if (l1_act !== l1_exp) begin
                    $display("    [ERROR] Layer 1 Neuron %0d mismatch! RTL=%0d, Exp=%0d", n, l1_act, l1_exp);
                    total_errors = total_errors + 1;
                end
            end

            // Layer 2 verification
            for (n = 0; n < LAYER2_NEURONS; n = n + 1) begin
                logic signed [7:0] l2_act;
                logic signed [7:0] l2_exp;
                l2_act = dut.layer2_output[n*8 +: 8];
                l2_exp = mem_expected_l2[s * LAYER2_NEURONS + n];
                if (l2_act !== l2_exp) begin
                    $display("    [ERROR] Layer 2 Neuron %0d mismatch! RTL=%0d, Exp=%0d", n, l2_act, l2_exp);
                    total_errors = total_errors + 1;
                end
            end

            // Check final class prediction
            if (predicted_class === mem_expected_classes[s]) begin
                classification_matches = classification_matches + 1;
                $display("  Status                : [PASSED]");
            end else begin
                total_errors = total_errors + 1;
                $display("  Status                : [FAILED]");
            end
            $display("------------------------------------------------------------");
        end

        // ----------------------------------------------------
        // 4. Print Final Verification Summary
        // ----------------------------------------------------
        $display("\n============================================================");
        $display("PHASE 7 REAL MODEL RTL VERIFICATION SUMMARY");
        $display("============================================================");
        $display("Total Test Samples     : %0d", TOTAL_SAMPLES);
        $display("Classification Matches : %0d / %0d", classification_matches, TOTAL_SAMPLES);
        $display("Total Bit Mismatches   : %0d", total_errors);

        if (total_errors == 0 && classification_matches == TOTAL_SAMPLES) begin
            $display("\n============================================================");
            $display(">>> ALL PHASE 7 VERIFICATION CHECKS PASSED (100%% PARITY) <<<");
            $display("============================================================");
        end else begin
            $display("\n============================================================");
            $display(">>> PHASE 7 VERIFICATION FAILED <<<");
            $display("============================================================");
        end

        $finish;
    end

endmodule
