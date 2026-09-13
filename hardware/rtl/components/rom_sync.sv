// ============================================================
// TinyNPU - Synchronous Parameter ROM Module
// ============================================================
//
// Synthesizable synchronous ROM module for storing neural
// network weights and biases on-chip.
//
// On FPGA, synthesis tools (Vivado/Quartus) automatically infer
// Block RAM (BRAM) or Distributed LUT-RAM from this template.
//
// Parameters:
//     WIDTH     - Bit-width of each entry (e.g. 8 for INT8, 32 for INT32)
//     DEPTH     - Total number of memory entries
//     INIT_FILE - Path to the hexadecimal initialization file (.mem)
// ============================================================

`timescale 1ns/1ps

module rom_sync #(
    parameter integer WIDTH     = 8,
    parameter integer DEPTH     = 294,
    parameter string  INIT_FILE = "data/mem/weights1.mem"
)(
    input  logic                     clk,
    input  logic [$clog2(DEPTH)-1:0] addr,
    output logic signed [WIDTH-1:0]  data_out
);

    // Memory array
    logic signed [WIDTH-1:0] memory [0:DEPTH-1];

    // Initialize ROM contents from .mem file
    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, memory);
        end
    end

    // Synchronous read (infers FPGA Block RAM / Distributed ROM)
    always_ff @(posedge clk) begin
        data_out <= memory[addr];
    end

endmodule
