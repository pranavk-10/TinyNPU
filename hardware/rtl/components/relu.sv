// ============================================================
// TinyNPU - ReLU Activation
// ============================================================
//
// ReLU:
//
//     ReLU(x) = max(0, x)
//
// Negative → 0
// Zero     → 0
// Positive → unchanged
//
// This implementation uses a continuous assignment instead
// of always_comb.
//
// This is extremely simple combinational hardware:
//
//     input_value
//          |
//          ▼
//       input < 0?
//        /     \
//      YES      NO
//       |        |
//       ▼        ▼
//       0     input_value
//        \      /
//         ▼    ▼
//          output
//
// ============================================================


module relu (

    // --------------------------------------------------------
    // Signed INT8 input.
    // --------------------------------------------------------

    input logic signed [7:0] input_value,

    // --------------------------------------------------------
    // Signed INT8 output.
    // --------------------------------------------------------

    output logic signed [7:0] output_value

);


    // --------------------------------------------------------
    // Continuous combinational assignment.
    //
    // If input_value is negative:
    //
    //     output = 0
    //
    // Otherwise:
    //
    //     output = input_value
    //
    // The ?: operator is synthesizable combinational logic.
    // --------------------------------------------------------

    assign output_value =
        (input_value < 0)
        ? 8'sd0
        : input_value;


endmodule