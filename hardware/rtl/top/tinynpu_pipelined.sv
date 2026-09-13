// ============================================================
// TinyNPU - High-Throughput Pipelined Streaming Accelerator (Phase 9)
// ============================================================
//
// Fully pipelined 3-stage streaming neural network engine:
//     - Stage 1: Layer 1 calculation (21 -> 14) + Requantize + ReLU
//     - Stage 2: Layer 2 calculation (14 -> 7) + Requantize + ReLU
//     - Stage 3: Layer 3 calculation (7 -> 1) + Requantize + Output
//
// Performance:
//     - Initiation Interval (II) = 1 (accepts 1 new sample every clock cycle)
//     - Latency = 3 clock cycles
//     - Throughput = 1 inference / clock cycle (100 MSamples/sec @ 100MHz)
// ============================================================

`timescale 1ns/1ps

module tinynpu_pipelined #(
    parameter integer INPUTS          = 21,
    parameter integer LAYER1_NEURONS  = 14,
    parameter integer LAYER2_NEURONS  = 7,
    parameter integer LAYER3_NEURONS  = 1,

    parameter string W1_MEM_FILE      = "data/mem/weights1.mem",
    parameter string B1_MEM_FILE      = "data/mem/bias1.mem",
    parameter string W2_MEM_FILE      = "data/mem/weights2.mem",
    parameter string B2_MEM_FILE      = "data/mem/bias2.mem",
    parameter string W3_MEM_FILE      = "data/mem/weights3.mem",
    parameter string B3_MEM_FILE      = "data/mem/bias3.mem",

    parameter integer L1_MULTIPLIER   = 1069,
    parameter integer L1_SHIFT        = 20,
    parameter integer L2_MULTIPLIER   = 4057,
    parameter integer L2_SHIFT        = 20,
    parameter integer L3_MULTIPLIER   = 419,
    parameter integer L3_SHIFT        = 20
)(
    input  logic                                     clk,
    input  logic                                     rst_n,

    // Streaming Input Interface
    input  logic                                     in_valid,
    input  logic signed [(INPUTS*8)-1:0]             inputs_bus,

    // Streaming Output Interface
    output logic                                     out_valid,
    output logic signed [7:0]                        output_value,
    output logic                                     predicted_class
);

    localparam integer W1_SIZE = INPUTS * LAYER1_NEURONS;
    localparam integer W2_SIZE = LAYER1_NEURONS * LAYER2_NEURONS;
    localparam integer W3_SIZE = LAYER2_NEURONS * LAYER3_NEURONS;

    // Weight and Bias ROM Arrays
    logic signed [7:0]  rom_w1 [0:W1_SIZE-1];
    logic signed [31:0] rom_b1 [0:LAYER1_NEURONS-1];
    logic signed [7:0]  rom_w2 [0:W2_SIZE-1];
    logic signed [31:0] rom_b2 [0:LAYER2_NEURONS-1];
    logic signed [7:0]  rom_w3 [0:W3_SIZE-1];
    logic signed [31:0] rom_b3 [0:LAYER3_NEURONS-1];

    initial begin
        $readmemh(W1_MEM_FILE, rom_w1);
        $readmemh(B1_MEM_FILE, rom_b1);
        $readmemh(W2_MEM_FILE, rom_w2);
        $readmemh(B2_MEM_FILE, rom_b2);
        $readmemh(W3_MEM_FILE, rom_w3);
        $readmemh(B3_MEM_FILE, rom_b3);
    end

    logic signed [(W1_SIZE*8)-1:0]          w1_packed;
    logic signed [(LAYER1_NEURONS*32)-1:0]  b1_packed;
    logic signed [(W2_SIZE*8)-1:0]          w2_packed;
    logic signed [(LAYER2_NEURONS*32)-1:0]  b2_packed;
    logic signed [(W3_SIZE*8)-1:0]          w3_packed;
    logic signed [31:0]                     b3_val;

    integer p;
    always_comb begin
        for (p = 0; p < W1_SIZE; p = p + 1)
            w1_packed[p*8 +: 8] = rom_w1[p];
        for (p = 0; p < LAYER1_NEURONS; p = p + 1)
            b1_packed[p*32 +: 32] = rom_b1[p];

        for (p = 0; p < W2_SIZE; p = p + 1)
            w2_packed[p*8 +: 8] = rom_w2[p];
        for (p = 0; p < LAYER2_NEURONS; p = p + 1)
            b2_packed[p*32 +: 32] = rom_b2[p];

        for (p = 0; p < W3_SIZE; p = p + 1)
            w3_packed[p*8 +: 8] = rom_w3[p];
        b3_val = rom_b3[0];
    end

    // Pipeline Registers
    logic                                     v1_reg, v2_reg, v3_reg;
    logic signed [(LAYER1_NEURONS*8)-1:0]     l1_reg;
    logic signed [(LAYER2_NEURONS*8)-1:0]     l2_reg;
    logic signed [7:0]                        l3_reg;

    // Combinational layer wires
    logic signed [(LAYER1_NEURONS*8)-1:0]     comb_l1;
    logic signed [(LAYER2_NEURONS*8)-1:0]     comb_l2;
    logic signed [7:0]                        comb_l3;

    // Layer 1: 21 -> 14
    layer1 #(
        .INPUTS(INPUTS),
        .NEURONS(LAYER1_NEURONS),
        .MULTIPLIER(L1_MULTIPLIER),
        .SHIFT(L1_SHIFT)
    ) u_layer1 (
        .inputs_bus(inputs_bus),
        .weights_bus(w1_packed),
        .biases_bus(b1_packed),
        .outputs_bus(comb_l1)
    );

    // Layer 2: 14 -> 7
    layer2 #(
        .INPUTS(LAYER1_NEURONS),
        .NEURONS(LAYER2_NEURONS),
        .MULTIPLIER(L2_MULTIPLIER),
        .SHIFT(L2_SHIFT)
    ) u_layer2 (
        .inputs_bus(l1_reg),
        .weights_bus(w2_packed),
        .biases_bus(b2_packed),
        .outputs_bus(comb_l2)
    );

    // Layer 3: 7 -> 1
    layer3 #(
        .INPUTS(LAYER2_NEURONS),
        .NEURONS(LAYER3_NEURONS),
        .MULTIPLIER(L3_MULTIPLIER),
        .SHIFT(L3_SHIFT)
    ) u_layer3 (
        .inputs_bus(l2_reg),
        .weights_bus(w3_packed),
        .bias(b3_val),
        .output_value(comb_l3)
    );

    // Pipeline Synchronous Update
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1_reg <= 1'b0;
            v2_reg <= 1'b0;
            v3_reg <= 1'b0;
            l1_reg <= '0;
            l2_reg <= '0;
            l3_reg <= '0;
        end else begin
            // Stage 1
            v1_reg <= in_valid;
            if (in_valid) begin
                l1_reg <= comb_l1;
            end

            // Stage 2
            v2_reg <= v1_reg;
            if (v1_reg) begin
                l2_reg <= comb_l2;
            end

            // Stage 3
            v3_reg <= v2_reg;
            if (v2_reg) begin
                l3_reg <= comb_l3;
            end
        end
    end

    assign out_valid       = v3_reg;
    assign output_value    = l3_reg;
    assign predicted_class = (l3_reg > 8'sd0) ? 1'b1 : 1'b0;

endmodule
