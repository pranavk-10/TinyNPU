# ============================================================
# TinyNPU FPGA Synthesis & Implementation Constraints (.xdc)
# Target Architecture: Xilinx 7-Series (Artix-7 / Spartan-7 / Zynq-7000)
# ============================================================

# Primary 100 MHz Clock Constraint (Period = 10.000 ns)
create_clock -period 10.000 -name sys_clk [get_ports clk]

# Input / Output Timing Constraints
set_input_delay -clock sys_clk -max 2.000 [get_ports {rst_n start inputs_bus[*]}]
set_input_delay -clock sys_clk -min 0.500 [get_ports {rst_n start inputs_bus[*]}]

set_output_delay -clock sys_clk -max 2.000 [get_ports {ready valid_out output_value[*] predicted_class latency_cycles[*]}]
set_output_delay -clock sys_clk -min 0.500 [get_ports {ready valid_out output_value[*] predicted_class latency_cycles[*]}]

# Optimization Directives
set_property DONT_TOUCH true [get_cells u_layer1]
set_property DONT_TOUCH true [get_cells u_layer2]
set_property DONT_TOUCH true [get_cells u_layer3]
