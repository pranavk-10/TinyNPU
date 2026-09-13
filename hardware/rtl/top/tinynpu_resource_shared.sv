// ============================================================
// TinyNPU - Ultra-Low-Area Resource-Shared NPU Engine (Phase 9)
// ============================================================
//
// Architecture:
//     - Uses 1 single time-multiplexed hardware MAC unit
//     - Sequentially iterates through:
//           Layer 1: 14 neurons × 21 MAC cycles
//           Layer 2:  7 neurons × 14 MAC cycles
//           Layer 3:  1 neuron  ×  7 MAC cycles
//     - Total execution time = 294 + 98 + 7 = 399 clock cycles per inference
//     - Multiplier count reduced from 399 down to 1 (99.7% area reduction)
//     - Ideal for resource-constrained edge FPGAs (Lattice iCE40, etc.)
// ============================================================

`timescale 1ns/1ps

module tinynpu_resource_shared #(
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

    input  logic                                     start,
    output logic                                     ready,
    output logic                                     valid_out,

    input  logic signed [(INPUTS*8)-1:0]             inputs_bus,
    output logic signed [7:0]                        output_value,
    output logic                                     predicted_class,
    output logic [31:0]                              total_cycles
);

    localparam integer W1_SIZE = INPUTS * LAYER1_NEURONS;         // 294
    localparam integer W2_SIZE = LAYER1_NEURONS * LAYER2_NEURONS; // 98
    localparam integer W3_SIZE = LAYER2_NEURONS * LAYER3_NEURONS; // 7

    // On-Chip Memories
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

    // Intermediate Activation Buffers
    logic signed [7:0] act_input [0:INPUTS-1];
    logic signed [7:0] act_l1    [0:LAYER1_NEURONS-1];
    logic signed [7:0] act_l2    [0:LAYER2_NEURONS-1];
    logic signed [7:0] act_l3;

    // Sequential State Machine
    typedef enum logic [2:0] {
        ST_IDLE      = 3'b000,
        ST_LOAD_IN   = 3'b001,
        ST_LAYER1    = 3'b010,
        ST_LAYER2    = 3'b011,
        ST_LAYER3    = 3'b100,
        ST_DONE      = 3'b101
    } state_e;

    state_e state;

    // Counters
    integer neuron_idx;
    integer input_idx;
    logic signed [31:0] acc;
    logic [31:0] cycle_ctr;

    // Single Requantize and ReLU instance function
    function automatic logic signed [7:0] requant_relu(
        input logic signed [31:0] in_acc,
        input integer mult,
        input integer shift
    );
        logic signed [63:0] scaled;
        logic signed [63:0] shifted;
        logic signed [7:0]  clamped;
        begin
            scaled  = in_acc * mult;
            shifted = scaled >>> shift;
            if (shifted > 127) clamped = 8'sd127;
            else if (shifted < -128) clamped = -8'sd128;
            else clamped = shifted[7:0];

            requant_relu = (clamped < 0) ? 8'sd0 : clamped;
        end
    endfunction

    integer i;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= ST_IDLE;
            neuron_idx  <= 0;
            input_idx   <= 0;
            acc         <= 0;
            cycle_ctr   <= 0;
            act_l3      <= 0;
        end else begin
            case (state)
                ST_IDLE: begin
                    cycle_ctr  <= 0;
                    neuron_idx <= 0;
                    input_idx  <= 0;
                    if (start) begin
                        for (i = 0; i < INPUTS; i = i + 1)
                            act_input[i] <= inputs_bus[i*8 +: 8];
                        state <= ST_LAYER1;
                        acc   <= rom_b1[0];
                    end
                end

                ST_LAYER1: begin
                    cycle_ctr <= cycle_ctr + 1;
                    acc <= acc + (act_input[input_idx] * rom_w1[neuron_idx * INPUTS + input_idx]);

                    if (input_idx == INPUTS - 1) begin
                        // Finish neuron
                        act_l1[neuron_idx] <= requant_relu(
                            acc + (act_input[input_idx] * rom_w1[neuron_idx * INPUTS + input_idx]),
                            L1_MULTIPLIER,
                            L1_SHIFT
                        );
                        input_idx <= 0;
                        if (neuron_idx == LAYER1_NEURONS - 1) begin
                            neuron_idx <= 0;
                            acc        <= rom_b2[0];
                            state      <= ST_LAYER2;
                        end else begin
                            neuron_idx <= neuron_idx + 1;
                            acc        <= rom_b1[neuron_idx + 1];
                        end
                    end else begin
                        input_idx <= input_idx + 1;
                    end
                end

                ST_LAYER2: begin
                    cycle_ctr <= cycle_ctr + 1;
                    acc <= acc + (act_l1[input_idx] * rom_w2[neuron_idx * LAYER1_NEURONS + input_idx]);

                    if (input_idx == LAYER1_NEURONS - 1) begin
                        act_l2[neuron_idx] <= requant_relu(
                            acc + (act_l1[input_idx] * rom_w2[neuron_idx * LAYER1_NEURONS + input_idx]),
                            L2_MULTIPLIER,
                            L2_SHIFT
                        );
                        input_idx <= 0;
                        if (neuron_idx == LAYER2_NEURONS - 1) begin
                            neuron_idx <= 0;
                            acc        <= rom_b3[0];
                            state      <= ST_LAYER3;
                        end else begin
                            neuron_idx <= neuron_idx + 1;
                            acc        <= rom_b2[neuron_idx + 1];
                        end
                    end else begin
                        input_idx <= input_idx + 1;
                    end
                end

                ST_LAYER3: begin
                    cycle_ctr <= cycle_ctr + 1;
                    acc <= acc + (act_l2[input_idx] * rom_w3[input_idx]);

                    if (input_idx == LAYER2_NEURONS - 1) begin
                        act_l3 <= requant_relu(
                            acc + (act_l2[input_idx] * rom_w3[input_idx]),
                            L3_MULTIPLIER,
                            L3_SHIFT
                        );
                        input_idx <= 0;
                        state     <= ST_DONE;
                    end else begin
                        input_idx <= input_idx + 1;
                    end
                end

                ST_DONE: begin
                    if (start) begin
                        for (i = 0; i < INPUTS; i = i + 1)
                            act_input[i] <= inputs_bus[i*8 +: 8];
                        state      <= ST_LAYER1;
                        neuron_idx <= 0;
                        input_idx  <= 0;
                        acc        <= rom_b1[0];
                        cycle_ctr  <= 0;
                    end else begin
                        state <= ST_IDLE;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    assign ready           = (state == ST_IDLE);
    assign valid_out       = (state == ST_DONE);
    assign output_value    = act_l3;
    assign predicted_class = (act_l3 > 8'sd0) ? 1'b1 : 1'b0;
    assign total_cycles    = cycle_ctr;

endmodule
