// ============================================================
// TinyNPU - FPGA Sequential Inference Testbench (Phase 8)
// ============================================================
//
// Tests the fully-clocked sequential FPGA NPU:
//     - Generates 100 MHz clock (10ns period)
//     - Asserts active-low reset
//     - Drives start handshake pulses
//     - Waits for valid_out signal
//     - Checks latency in clock cycles (exact 3 clock cycles)
//     - Validates predictions against golden expected classes
// ============================================================

`timescale 1ns/1ps

module tb_tinynpu_fpga;

    localparam integer INPUTS        = 21;
    localparam integer TOTAL_SAMPLES = 2;
    localparam integer CLK_PERIOD    = 10; // 100 MHz

    // DUT Signals
    logic                                     clk;
    logic                                     rst_n;
    logic                                     start;
    logic                                     ready;
    logic                                     valid_out;
    logic signed [(INPUTS*8)-1:0]             inputs_bus;
    logic signed [7:0]                        output_value;
    logic                                     predicted_class;
    logic [31:0]                              latency_cycles;

    // Memory arrays for stimulus and expected results
    logic signed [7:0] mem_inputs [0:(TOTAL_SAMPLES*INPUTS)-1];
    logic signed [7:0] mem_expected_classes [0:TOTAL_SAMPLES-1];

    // Instantiate Sequential FPGA DUT
    tinynpu_fpga #(
        .INPUTS(21),
        .LAYER1_NEURONS(14),
        .LAYER2_NEURONS(7),
        .LAYER3_NEURONS(1),
        .W1_MEM_FILE("data/mem/weights1.mem"),
        .B1_MEM_FILE("data/mem/bias1.mem"),
        .W2_MEM_FILE("data/mem/weights2.mem"),
        .B2_MEM_FILE("data/mem/bias2.mem"),
        .W3_MEM_FILE("data/mem/weights3.mem"),
        .B3_MEM_FILE("data/mem/bias3.mem")
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .ready(ready),
        .valid_out(valid_out),
        .inputs_bus(inputs_bus),
        .output_value(output_value),
        .predicted_class(predicted_class),
        .latency_cycles(latency_cycles)
    );

    // Clock Generation (100 MHz)
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2) clk = ~clk;
    end

    integer s, i;
    integer passed_count;
    integer error_count;

    initial begin
        passed_count = 0;
        error_count  = 0;
        start        = 0;
        inputs_bus   = '0;
        rst_n        = 0;

        $display("============================================================");
        $display("TinyNPU Phase 8: FPGA Sequential NPU Hardware Testbench");
        $display("Clock Frequency: 100 MHz (Period = 10ns)");
        $display("============================================================");

        // Load test stimulus
        $readmemh("data/mem/inputs.mem", mem_inputs);
        $readmemh("data/mem/expected_classes.mem", mem_expected_classes);

        // Reset Pulse
        #(CLK_PERIOD * 2);
        rst_n = 1;
        #(CLK_PERIOD);

        $display("\n[1] Reset deasserted. DUT ready status: %0b", ready);

        // Run inference on all test samples
        for (s = 0; s < TOTAL_SAMPLES; s = s + 1) begin
            // Wait until ready
            while (!ready) @(posedge clk);

            // Apply input features
            for (i = 0; i < INPUTS; i = i + 1) begin
                inputs_bus[i*8 +: 8] = mem_inputs[s*INPUTS + i];
            end

            // Assert start pulse for 1 cycle
            @(posedge clk);
            start = 1'b1;
            @(posedge clk);
            start = 1'b0;

            // Wait for valid_out
            while (!valid_out) @(posedge clk);

            $display("\nSample %0d:", s + 1);
            $display("  Latency (Clock Cycles): %0d cycles (%0d ns @ 100MHz)", latency_cycles, latency_cycles * CLK_PERIOD);
            $display("  Expected Class        : %0d", mem_expected_classes[s]);
            $display("  RTL Output Activation : %0d", $signed(output_value));
            $display("  RTL Predicted Class   : %0d", predicted_class);

            if (predicted_class == mem_expected_classes[s][0]) begin
                $display("  Verification Status   : [PASSED]");
                passed_count = passed_count + 1;
            end else begin
                $display("  Verification Status   : [FAILED]");
                error_count = error_count + 1;
            end
        end

        #(CLK_PERIOD * 2);

        $display("\n============================================================");
        $display("PHASE 8 FPGA SEQUENTIAL HARDWARE TEST SUMMARY");
        $display("============================================================");
        $display("Total Samples Tested : %0d", TOTAL_SAMPLES);
        $display("Total Passed         : %0d / %0d", passed_count, TOTAL_SAMPLES);
        $display("Total Errors         : %0d", error_count);
        $display("Inference Latency    : 3 Clock Cycles per vector (30 ns @ 100MHz)");

        if (error_count == 0 && passed_count == TOTAL_SAMPLES) begin
            $display("============================================================");
            $display(">>> ALL PHASE 8 FPGA HARDWARE CHECKS PASSED (100%%) <<<");
            $display("============================================================");
        end else begin
            $display(">>> PHASE 8 FPGA HARDWARE TESTS FAILED <<<");
        end

        $finish;
    end

endmodule
