// ============================================================
// TinyNPU - MAC Testbench
// ============================================================
//
// This testbench verifies the MAC independently.
//
// We will test:
//
//     10 × 5
//      3 × 2
//     -4 × 7
//
// with a bias of:
//
//     10
//
// Expected:
//
//     10
//     + 50
//     + 6
//     - 28
//     = 38
//
// Therefore:
//
//     RTL result = 38
//
// ============================================================


module tb_mac;


    // --------------------------------------------------------
    // Number of input/weight pairs.
    // --------------------------------------------------------

    localparam integer N = 3;


    // --------------------------------------------------------
    // Input array.
    //
    // Three signed INT8 values.
    // --------------------------------------------------------

    logic signed [7:0] inputs [0:N-1];


    // --------------------------------------------------------
    // Weight array.
    //
    // Three signed INT8 values.
    // --------------------------------------------------------

    logic signed [7:0] weights [0:N-1];


    // --------------------------------------------------------
    // INT32 bias.
    // --------------------------------------------------------

    logic signed [31:0] bias;


    // --------------------------------------------------------
    // Result produced by the MAC.
    // --------------------------------------------------------

    logic signed [31:0] result;


    // --------------------------------------------------------
    // Instantiate the MAC module.
    //
    // N = 3 because this test uses three inputs.
    // --------------------------------------------------------

    mac #(
        .N(N)
    ) dut (

        .inputs(inputs),

        .weights(weights),

        .bias(bias),

        .result(result)

    );


    // --------------------------------------------------------
    // Test sequence.
    // --------------------------------------------------------

    initial begin


        // Set input 0 = 10.
        inputs[0] = 8'sd10;


        // Set input 1 = 3.
        inputs[1] = 8'sd3;


        // Set input 2 = -4.
        inputs[2] = -8'sd4;


        // Set weight 0 = 5.
        weights[0] = 8'sd5;


        // Set weight 1 = 2.
        weights[1] = 8'sd2;


        // Set weight 2 = 7.
        weights[2] = 8'sd7;


        // Set bias = 10.
        bias = 32'sd10;


        // Wait one simulation time unit.
        //
        // This gives the combinational logic time
        // to calculate the result.
        #1;


        // Print the result.
        $display("====================================");
        $display("TinyNPU MAC Test");
        $display("====================================");


        $display(
            "Input 0 = %0d, Weight 0 = %0d",
            inputs[0],
            weights[0]
        );


        $display(
            "Input 1 = %0d, Weight 1 = %0d",
            inputs[1],
            weights[1]
        );


        $display(
            "Input 2 = %0d, Weight 2 = %0d",
            inputs[2],
            weights[2]
        );


        $display(
            "Bias = %0d",
            bias
        );


        $display(
            "Expected result = 38"
        );


        $display(
            "RTL result      = %0d",
            result
        );


        // Compare RTL result with expected result.
        if (result == 32'sd38) begin


            // Result matches.
            $display("TEST PASSED");


        end
        else begin


            // Result does not match.
            $display("TEST FAILED");


        end


        // End the simulation.
        $finish;


    end


endmodule