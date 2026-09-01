// ============================================================
// TinyNPU - Requantization Testbench
// ============================================================
//
// This testbench verifies:
//
//     INT32 accumulator
//             ↓
//       requantization
//             ↓
//          INT8
//
// Using the layer-1 scale ratio.
//
// ============================================================


module tb_requantize;


    // --------------------------------------------------------
    // Test input.
    //
    // This represents an INT32 MAC accumulator.
    // --------------------------------------------------------

    logic signed [31:0] accumulator;


    // --------------------------------------------------------
    // Output produced by the requantizer.
    // --------------------------------------------------------

    logic signed [7:0] output_value;


    // --------------------------------------------------------
    // Instantiate the requantizer.
    //
    // These constants approximate the layer-1 scale ratio.
    //
    // MULTIPLIER / 2^SHIFT
    //
    // = 1223 / 2^20
    //
    // ≈ 0.00116784
    //
    // which is close to the required layer-1 ratio.
    // --------------------------------------------------------

    requantize #(

        .MULTIPLIER(1223),

        .SHIFT(20)

    ) dut (

        .accumulator(accumulator),

        .output_value(output_value)

    );


    // --------------------------------------------------------
    // Test sequence.
    // --------------------------------------------------------

    initial begin


        $display("====================================");
        $display("TinyNPU Requantization Test");
        $display("====================================");


        // ====================================================
        // TEST 1
        // ====================================================
        //
        // Accumulator = 0
        //
        // Expected output = 0
        // ====================================================

        accumulator = 32'sd0;

        #1;


        $display(
            "Accumulator = %0d | Output = %0d",
            accumulator,
            output_value
        );


        if (output_value == 8'sd0) begin

            $display("TEST 1 PASSED");

        end
        else begin

            $display("TEST 1 FAILED");

        end


        // ====================================================
        // TEST 2
        // ====================================================
        //
        // Positive accumulator.
        //
        // Expected:
        //
        //     10000 × 0.00116784
        //     ≈ 11.67
        //
        // Integer result ≈ 11.
        // ====================================================

        accumulator = 32'sd10000;

        #1;


        $display(
            "Accumulator = %0d | Output = %0d",
            accumulator,
            output_value
        );


        if (
            output_value >= 8'sd10 &&
            output_value <= 8'sd13
        ) begin

            $display("TEST 2 PASSED");

        end
        else begin

            $display("TEST 2 FAILED");

        end


        // ====================================================
        // TEST 3
        // ====================================================
        //
        // Negative accumulator.
        //
        // Expected:
        //
        //     -10000 × 0.00116784
        //     ≈ -11.67
        //
        // ====================================================

        accumulator = -32'sd10000;

        #1;


        $display(
            "Accumulator = %0d | Output = %0d",
            accumulator,
            output_value
        );


        if (
            output_value <= -8'sd10 &&
            output_value >= -8'sd13
        ) begin

            $display("TEST 3 PASSED");

        end
        else begin

            $display("TEST 3 FAILED");

        end


        // ====================================================
        // TEST 4
        // ====================================================
        //
        // Large positive value.
        //
        // This should saturate at +127 if necessary.
        // ====================================================

        accumulator = 32'sd1000000;

        #1;


        $display(
            "Accumulator = %0d | Output = %0d",
            accumulator,
            output_value
        );


        if (output_value <= 8'sd127) begin

            $display("TEST 4 PASSED");

        end
        else begin

            $display("TEST 4 FAILED");

        end


        // ====================================================
        // TEST 5
        // ====================================================
        //
        // Large negative value.
        //
        // This should saturate at -128 if necessary.
        // ====================================================

        accumulator = -32'sd1000000;

        #1;


        $display(
            "Accumulator = %0d | Output = %0d",
            accumulator,
            output_value
        );


        if (output_value >= -8'sd128) begin

            $display("TEST 5 PASSED");

        end
        else begin

            $display("TEST 5 FAILED");

        end


        // ====================================================
        // End simulation.
        // ====================================================

        $display("====================================");
        $display("Requantization Test Complete");
        $display("====================================");


        $finish;


    end


endmodule