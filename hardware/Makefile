# TinyNPU Hardware Simulation Makefile
IVERILOG = iverilog
VVP = vvp
FLAGS = -g2012 -I..

RTL_BASE_SRCS = rtl/components/mac.sv \
                rtl/components/relu.sv \
                rtl/components/requantize.sv \
                rtl/components/rom_sync.sv \
                rtl/layers/neuron.sv \
                rtl/layers/layer1.sv \
                rtl/layers/layer2.sv \
                rtl/layers/layer3.sv

.PHONY: all real_sim fpga_sim shared_sim top_sim clean all_sim

all: all_sim

sim_dir:
	mkdir -p sim

# Phase 7: Combinational Real-Model Verification
real_sim: sim_dir
	$(IVERILOG) $(FLAGS) -s tb_tinynpu_real -o sim/tinynpu_real_sim.vvp $(RTL_BASE_SRCS) rtl/top/tinynpu.sv rtl/testbenches/tb_tinynpu_real.sv
	$(VVP) sim/tinynpu_real_sim.vvp

# Phase 8: Sequential Clocked FPGA NPU Simulation
fpga_sim: sim_dir
	$(IVERILOG) $(FLAGS) -s tb_tinynpu_fpga -o sim/tinynpu_fpga_sim.vvp $(RTL_BASE_SRCS) rtl/top/tinynpu_fpga.sv rtl/testbenches/tb_tinynpu_fpga.sv
	$(VVP) sim/tinynpu_fpga_sim.vvp

# Phase 9: Resource-Shared 1-MAC Engine Simulation
shared_sim: sim_dir
	$(IVERILOG) $(FLAGS) -s tb_tinynpu_resource_shared -o sim/tinynpu_shared_sim.vvp $(RTL_BASE_SRCS) rtl/top/tinynpu_resource_shared.sv rtl/testbenches/tb_tinynpu_resource_shared.sv
	$(VVP) sim/tinynpu_shared_sim.vvp

# Top Dummy Integration Test
top_sim: sim_dir
	$(IVERILOG) $(FLAGS) -s tb_tinynpu -o sim/tinynpu_sim.vvp $(RTL_BASE_SRCS) rtl/top/tinynpu.sv rtl/testbenches/tb_tinynpu.sv
	$(VVP) sim/tinynpu_sim.vvp

# Run All Verification Testbenches
all_sim: real_sim fpga_sim shared_sim
	@echo "============================================================"
	@echo ">>> ALL HARDWARE SIMULATION TARGETS PASSED SUCCESSFULLY! <<<"
	@echo "============================================================"

clean:
	rm -rf sim
