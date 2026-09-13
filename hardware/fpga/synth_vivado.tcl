# ============================================================
# TinyNPU - Automated Xilinx Vivado Non-Project Batch Synthesis
# ============================================================
# Usage:
#   vivado -mode batch -source hardware/fpga/synth_vivado.tcl
# ============================================================

set PROJECT_ROOT [file normalize [file dirname [info script]]/../..]
set RTL_DIR      $PROJECT_ROOT/hardware/rtl
set FPGA_DIR     $PROJECT_ROOT/hardware/fpga
set REPORTS_DIR  $PROJECT_ROOT/hardware/fpga/reports

file mkdir $REPORTS_DIR

puts "============================================================"
puts "TinyNPU: Starting Vivado Synthesis & Timing Closure Flow"
puts "============================================================"

# Set Target FPGA Device (Xilinx Artix-7: xc7a35tcsg324-1)
set_part xc7a35tcsg324-1

# Read SystemVerilog Source Files
read_verilog -sv [list \
    $RTL_DIR/components/mac.sv \
    $RTL_DIR/components/relu.sv \
    $RTL_DIR/components/requantize.sv \
    $RTL_DIR/components/rom_sync.sv \
    $RTL_DIR/layers/neuron.sv \
    $RTL_DIR/layers/layer1.sv \
    $RTL_DIR/layers/layer2.sv \
    $RTL_DIR/layers/layer3.sv \
    $RTL_DIR/top/tinynpu_fpga.sv \
]

# Read XDC Constraints
read_xdc $FPGA_DIR/tinynpu.xdc

# Run RTL Synthesis
puts "--- Running Synthesis ---"
synth_design -top tinynpu_fpga -part xc7a35tcsg324-1 -include_dirs [list $PROJECT_ROOT]

# Optimization & Placement
puts "--- Running Logic Optimization & Placement ---"
opt_design
place_design
phys_opt_design

# Routing
puts "--- Running Routing ---"
route_design

# Generate Comprehensive Reports
puts "--- Generating Reports ---"
report_utilization -file $REPORTS_DIR/utilization_placed.rpt
report_timing_summary -file $REPORTS_DIR/timing_summary.rpt
report_power -file $REPORTS_DIR/power_analysis.rpt
report_datasheet -file $REPORTS_DIR/datasheet.rpt

puts "============================================================"
puts "TinyNPU: Vivado Implementation Flow Complete!"
puts "Reports written to: $REPORTS_DIR"
puts "============================================================"
exit
