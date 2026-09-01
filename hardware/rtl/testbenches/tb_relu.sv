// ============================================================
// TinyNPU - ReLU Testbench
// ============================================================
//
// This testbench checks several ReLU cases:
//
//     -10 → 0
//      -1 → 0
//       0 → 0
//       5 → 5
//     127 → 127
//
// ============================================================


module tb_relu;


    // --------------------------------------------------------
    // Signal connected to the ReLU input.
    // --------------------------------------------------------

    logic signed [7:0] input_value;


    // --------------------------------------------------------
    // Signal connected to the ReLU output.
    // --------------------------------------------------------

    logic signed [7:0] output_value;


    // --------------------------------------------------------
    // Instantiate the ReLU module.
    //
    // "dut" means Device Under Test.
    // --------------------------------------------------------

    relu dut (

        // Connect testbench input to ReLU input.
        .input_value(input_value),

        // Connect ReLU output to testbench output.
        .output_value(output_value)

    );


    // --------------------------------------------------------
    // Test sequence.
    // --------------------------------------------------------

    initial begin


        $display("====================================");
        $display("TinyNPU ReLU Test");
        $display("====================================");


        // ====================================================
        // TEST 1
        // ====================================================
        //
        // Input = -10
        // Expected = 0
        // ====================================================

        input_value = -8'sd10;

        // Give combinational logic time to respond.
        #1;

        $display(
            "Input = %0d | Expected = 0 | RTL = %0d",
            input_value,
            output_value
        );

        // Check expected result.
        if (output_value != 8'sd0) begin

            $display("TEST 1 FAILED");

        end
        else begin

            $display("TEST 1 PASSED");

        end


        // ====================================================
        // TEST 2
        // ====================================================
        //
        // Input = -1
        // Expected = 0
        // ====================================================

        input_value = -8'sd1;

        #1;

        $display(
            "Input = %0d | Expected = 0 | RTL = %0d",
            input_value,
            output_value
        );

        if (output_value != 8'sd0) begin

            $display("TEST 2 FAILED");

        end
        else begin

            $display("TEST 2 PASSED");

        end


        // ====================================================
        // TEST 3
        // ====================================================
        //
        // Input = 0
        // Expected = 0
        // ====================================================

        input_value = 8'sd0;

        #1;

        $display(
            "Input = %0d | Expected = 0 | RTL = %0d",
            input_value,
            output_value
        );

        if (output_value != 8'sd0) begin

            $display("TEST 3 FAILED");

        end
        else begin

            $display("TEST 3 PASSED");

        end


        // ====================================================
        // TEST 4
        // ====================================================
        //
        // Input = 5
        // Expected = 5
        // ====================================================

        input_value = 8'sd5;

        #1;

        $display(
            "Input = %0d | Expected = 5 | RTL = %0d",
            input_value,
            output_value
        );

        if (output_value != 8'sd5) begin

            $display("TEST 4 FAILED");

        end
        else begin

            $display("TEST 4 PASSED");

        end


        // ====================================================
        // TEST 5
        // ====================================================
        //
        // Input = 127
        // Expected = 127
        // ====================================================

        input_value = 8'sd127;

        #1;

        $display(
            "Input = %0d | Expected = 127 | RTL = %0d",
            input_value,
            output_value
        );

        if (output_value != 8'sd127) begin

            $display("TEST 5 FAILED");

        end
        else begin

            $display("TEST 5 PASSED");

        end


        // ====================================================
        // Finish simulation.
        // ====================================================

        $display("====================================");
        $display("ReLU Test Complete");
        $display("====================================");

        $finish;


    end


endmodule