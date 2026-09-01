// ============================================================
// TinyNPU - Single Neuron Testbench
// ============================================================
//
// We test one complete neuron:
//
//     inputs
//        ↓
//       MAC
//        ↓
//     INT32
//        ↓
//   Requantize
//        ↓
//      INT8
//        ↓
//      ReLU
//        ↓
//     output
//
// We use:
//
//     Input 0  = 10
//     Input 1  = 3
//     Input 2  = -4
//
//     Weight 0 = 5
//     Weight 1 = 2
//     Weight 2 = 7
//
//     Bias = 10
//
// MAC:
//
//     (10 × 5) + (3 × 2) + (-4 × 7) + 10
//
//     = 50 + 6 - 28 + 10
//
//     = 38
//
// Requantization:
//
//     38 × 1223 / 2^20
//
//     ≈ 0
//
// Therefore:
//
//     ReLU(0) = 0
//
// ============================================================


module tb_neuron;


    // --------------------------------------------------------
    // Number of inputs.
    //
    // We use three inputs for this simple test.
    // --------------------------------------------------------

    localparam integer N = 3;


    // --------------------------------------------------------
    // Test inputs.
    // --------------------------------------------------------

    logic signed [7:0] inputs [0:N-1];


    // --------------------------------------------------------
    // Test weights.
    // --------------------------------------------------------

    logic signed [7:0] weights [0:N-1];


    // --------------------------------------------------------
    // Bias.
    // --------------------------------------------------------

    logic signed [31:0] bias;


    // --------------------------------------------------------
    // Output produced by the neuron.
    // --------------------------------------------------------

    logic signed [7:0] output_value;


    // --------------------------------------------------------
    // Instantiate the neuron.
    // --------------------------------------------------------

    neuron #(

        // Tell the neuron there are three inputs.
        .N(N),

        // Use the same multiplier tested previously.
        .MULTIPLIER(1223),

        // Use the same shift tested previously.
        .SHIFT(20)

    ) dut (

        // Connect test inputs.
        .inputs(inputs),

        // Connect test weights.
        .weights(weights),

        // Connect bias.
        .bias(bias),

        // Connect neuron output.
        .output_value(output_value)

    );


    // --------------------------------------------------------
    // Test sequence.
    // --------------------------------------------------------

    initial begin


        // ----------------------------------------------------
        // Test input 0.
        // ----------------------------------------------------

        inputs[0] = 8'sd10;


        // ----------------------------------------------------
        // Test input 1.
        // ----------------------------------------------------

        inputs[1] = 8'sd3;


        // ----------------------------------------------------
        // Test input 2.
        // ----------------------------------------------------

        inputs[2] = -8'sd4;


        // ----------------------------------------------------
        // Test weight 0.
        // ----------------------------------------------------

        weights[0] = 8'sd5;


        // ----------------------------------------------------
        // Test weight 1.
        // ----------------------------------------------------

        weights[1] = 8'sd2;


        // ----------------------------------------------------
        // Test weight 2.
        // ----------------------------------------------------

        weights[2] = 8'sd7;


        // ----------------------------------------------------
        // Set bias.
        // ----------------------------------------------------

        bias = 32'sd10;


        // ----------------------------------------------------
        // Wait for combinational logic to calculate.
        // ----------------------------------------------------

        #1;


        // ----------------------------------------------------
        // Display test information.
        // ----------------------------------------------------

        $display("====================================");
        $display("TinyNPU Single Neuron Test");
        $display("====================================");


        // Display the MAC result.
        $display(
            "MAC result       = %0d",
            dut.mac_result
        );


        // Display the requantized value.
        $display(
            "Requantized      = %0d",
            dut.requantized_value
        );


        // Display final ReLU output.
        $display(
            "Neuron output    = %0d",
            output_value
        );


        // ----------------------------------------------------
        // The MAC result should be 38.
        // ----------------------------------------------------

        if (dut.mac_result == 32'sd38) begin

            $display("MAC CHECK PASSED");

        end
        else begin

            $display("MAC CHECK FAILED");

        end


        // ----------------------------------------------------
        // For this small accumulator, the requantized result
        // should be zero.
        // ----------------------------------------------------

        if (dut.requantized_value == 8'sd0) begin

            $display("REQUANTIZATION CHECK PASSED");

        end
        else begin

            $display("REQUANTIZATION CHECK FAILED");

        end


        // ----------------------------------------------------
        // ReLU should therefore also produce zero.
        // ----------------------------------------------------

        if (output_value == 8'sd0) begin

            $display("RELU CHECK PASSED");

        end
        else begin

            $display("RELU CHECK FAILED");

        end


        // ----------------------------------------------------
        // Final test result.
        // ----------------------------------------------------

        if (
            dut.mac_result == 32'sd38 &&
            dut.requantized_value == 8'sd0 &&
            output_value == 8'sd0
        ) begin

            $display("====================================");
            $display("NEURON TEST PASSED");
            $display("====================================");

        end
        else begin

            $display("====================================");
            $display("NEURON TEST FAILED");
            $display("====================================");

        end


        // ----------------------------------------------------
        // End simulation.
        // ----------------------------------------------------

        $finish;


    end


endmodule