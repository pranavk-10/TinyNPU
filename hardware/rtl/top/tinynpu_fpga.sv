// ============================================================
// TinyNPU - FPGA Sequential Inference Accelerator (Phase 8)
// ============================================================
//
// Fully clocked sequential neural network accelerator with:
//
//     - Synchronous clock (clk) and active-low reset (rst_n)
//     - Standard Handshake Control FSM (start, ready, valid_out)
//     - Internal Parameter ROMs initialized from .mem files via $readmemh
//     - Layer Isolation Pipeline Registers for High-Frequency FPGA Clocking
//     - Hardware Cycle Counter for Exact Execution Latency Profiling
//
// Network: 21 (Input) -> 14 (Hidden 1) -> 7 (Hidden 2) -> 1 (Output)
// Total Trainable Parameters: 421 (399 INT8 weights + 22 INT32 biases)
// ============================================================

`timescale 1ns/1ps

module tinynpu_fpga #(
    parameter integer INPUTS          = 21,
    parameter integer LAYER1_NEURONS  = 14,
    parameter integer LAYER2_NEURONS  = 7,
    parameter integer LAYER3_NEURONS  = 1,

    // File paths for parameter initialization
    parameter string W1_MEM_FILE      = "data/mem/weights1.mem",
    parameter string B1_MEM_FILE      = "data/mem/bias1.mem",
    parameter string W2_MEM_FILE      = "data/mem/weights2.mem",
    parameter string B2_MEM_FILE      = "data/mem/bias2.mem",
    parameter string W3_MEM_FILE      = "data/mem/weights3.mem",
    parameter string B3_MEM_FILE      = "data/mem/bias3.mem",

    // Default Fixed-Point Requantization Parameters (overridable)
    parameter integer L1_MULTIPLIER   = 1069,
    parameter integer L1_SHIFT        = 20,
    parameter integer L2_MULTIPLIER   = 4057,
    parameter integer L2_SHIFT        = 20,
    parameter integer L3_MULTIPLIER   = 419,
    parameter integer L3_SHIFT        = 20
)(
    // Clock and Reset
    input  logic                                     clk,
    input  logic                                     rst_n,

    // Handshake Control Interface
    input  logic                                     start,
    output logic                                     ready,
    output logic                                     valid_out,

    // Input Feature Bus (21 INT8 features = 168 bits)
    input  logic signed [(INPUTS*8)-1:0]             inputs_bus,

    // Output Prediction Interface
    output logic signed [7:0]                        output_value,
    output logic                                     predicted_class,

    // Diagnostic Performance Counter
    output logic [31:0]                              latency_cycles
);

    // ========================================================
    // INTERNAL CONSTANTS & SIZES
    // ========================================================
    localparam integer W1_SIZE = INPUTS * LAYER1_NEURONS;           // 294
    localparam integer W2_SIZE = LAYER1_NEURONS * LAYER2_NEURONS;   // 98
    localparam integer W3_SIZE = LAYER2_NEURONS * LAYER3_NEURONS;   // 7

    // ========================================================
    // ON-CHIP PARAMETER STORAGE ARRAYS ($readmemh)
    // ========================================================
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

    // Flat packed buses from ROM arrays
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

    // ========================================================
    // SEQUENTIAL CONTROL FINITE STATE MACHINE (FSM)
    // ========================================================
    typedef enum logic [2:0] {
        STATE_IDLE        = 3'b000,
        STATE_LAYER1_EXEC = 3'b001,
        STATE_LAYER2_EXEC = 3'b010,
        STATE_LAYER3_EXEC = 3'b011,
        STATE_DONE        = 3'b100
    } state_t;

    state_t current_state, next_state;

    // Pipeline Registers
    logic signed [(INPUTS*8)-1:0]            reg_inputs;
    logic signed [(LAYER1_NEURONS*8)-1:0]    reg_layer1_out;
    logic signed [(LAYER2_NEURONS*8)-1:0]    reg_layer2_out;
    logic signed [7:0]                       reg_layer3_out;
    logic [31:0]                             reg_cycles;

    // Combinational layer output wires
    logic signed [(LAYER1_NEURONS*8)-1:0]    comb_layer1_out;
    logic signed [(LAYER2_NEURONS*8)-1:0]    comb_layer2_out;
    logic signed [7:0]                       comb_layer3_out;

    // ========================================================
    // LAYER INSTANTIATIONS
    // ========================================================

    // Layer 1: 21 -> 14
    layer1 #(
        .INPUTS(INPUTS),
        .NEURONS(LAYER1_NEURONS),
        .MULTIPLIER(L1_MULTIPLIER),
        .SHIFT(L1_SHIFT)
    ) u_layer1 (
        .inputs_bus(reg_inputs),
        .weights_bus(w1_packed),
        .biases_bus(b1_packed),
        .outputs_bus(comb_layer1_out)
    );

    // Layer 2: 14 -> 7
    layer2 #(
        .INPUTS(LAYER1_NEURONS),
        .NEURONS(LAYER2_NEURONS),
        .MULTIPLIER(L2_MULTIPLIER),
        .SHIFT(L2_SHIFT)
    ) u_layer2 (
        .inputs_bus(reg_layer1_out),
        .weights_bus(w2_packed),
        .biases_bus(b2_packed),
        .outputs_bus(comb_layer2_out)
    );

    // Layer 3: 7 -> 1
    layer3 #(
        .INPUTS(LAYER2_NEURONS),
        .NEURONS(LAYER3_NEURONS),
        .MULTIPLIER(L3_MULTIPLIER),
        .SHIFT(L3_SHIFT)
    ) u_layer3 (
        .inputs_bus(reg_layer2_out),
        .weights_bus(w3_packed),
        .bias(b3_val),
        .output_value(comb_layer3_out)
    );

    // ========================================================
    // STATE REGISTER & SEQUENTIAL DATAPATH
    // ========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state   <= STATE_IDLE;
            reg_inputs      <= '0;
            reg_layer1_out  <= '0;
            reg_layer2_out  <= '0;
            reg_layer3_out  <= '0;
            reg_cycles      <= '0;
        end else begin
            current_state   <= next_state;

            case (current_state)
                STATE_IDLE: begin
                    reg_cycles <= 32'd0;
                    if (start) begin
                        reg_inputs <= inputs_bus;
                    end
                end

                STATE_LAYER1_EXEC: begin
                    reg_cycles     <= reg_cycles + 1;
                    reg_layer1_out <= comb_layer1_out;
                end

                STATE_LAYER2_EXEC: begin
                    reg_cycles     <= reg_cycles + 1;
                    reg_layer2_out <= comb_layer2_out;
                end

                STATE_LAYER3_EXEC: begin
                    reg_cycles     <= reg_cycles + 1;
                    reg_layer3_out <= comb_layer3_out;
                end

                STATE_DONE: begin
                    // Hold result until next start
                end

                default: begin
                    reg_cycles <= 32'd0;
                end
            endcase
        end
    end

    // ========================================================
    // NEXT STATE LOGIC
    // ========================================================
    always_comb begin
        next_state = current_state;

        case (current_state)
            STATE_IDLE: begin
                if (start)
                    next_state = STATE_LAYER1_EXEC;
                else
                    next_state = STATE_IDLE;
            end

            STATE_LAYER1_EXEC: begin
                next_state = STATE_LAYER2_EXEC;
            end

            STATE_LAYER2_EXEC: begin
                next_state = STATE_LAYER3_EXEC;
            end

            STATE_LAYER3_EXEC: begin
                next_state = STATE_DONE;
            end

            STATE_DONE: begin
                if (start)
                    next_state = STATE_LAYER1_EXEC;
                else
                    next_state = STATE_IDLE;
            end

            default: next_state = STATE_IDLE;
        endcase
    end

    // ========================================================
    // OUTPUT ASSIGNMENTS
    // ========================================================
    assign ready           = (current_state == STATE_IDLE);
    assign valid_out       = (current_state == STATE_DONE);
    assign output_value    = reg_layer3_out;
    assign predicted_class = (reg_layer3_out > 8'sd0) ? 1'b1 : 1'b0;
    assign latency_cycles  = reg_cycles;

endmodule
