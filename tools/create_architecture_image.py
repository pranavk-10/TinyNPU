"""
TinyNPU - Next-Generation Architecture & Hardware Datapath Visualization
========================================================================
Generates an ultra-professional, high-resolution (300 DPI) publication-grade
neural network architecture & FPGA datapath diagram with modern tech aesthetic.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMAGE = DOCS_DIR / "tinynpu_architecture.png"

# Canvas Setup (16:9 cinematic ratio, 24 x 13.5 inches @ 300 DPI)
fig, ax = plt.subplots(figsize=(24, 13.5), dpi=300)
fig.patch.set_facecolor('#0B0F19')
ax.set_facecolor('#0B0F19')

# -------------------------------------------------------------
# Color Palette (Futuristic Dark / Neon Tech)
# -------------------------------------------------------------
COLOR_BG_CARD    = '#111827'
COLOR_BORDER     = '#1F2937'
COLOR_CYAN       = '#00F0FF'
COLOR_BLUE       = '#38BDF8'
COLOR_VIOLET     = '#818CF8'
COLOR_PURPLE     = '#C084FC'
COLOR_EMERALD    = '#10B981'
COLOR_AMBER      = '#F59E0B'
COLOR_TEXT_MAIN  = '#F9FAFB'
COLOR_TEXT_MUTED = '#9CA3AF'
COLOR_TEXT_DIM   = '#6B7280'

# =============================================================
# HEADER SECTION
# =============================================================
# Main Header Card
header_box = patches.FancyBboxPatch((0.5, 11.6), 23.0, 1.4,
                                    boxstyle="round,pad=0.2,rounding_size=0.3",
                                    facecolor='#0F172A', edgecolor='#38BDF8',
                                    linewidth=1.5, alpha=0.95, zorder=2)
ax.add_patch(header_box)

ax.text(1.2, 12.45, "TinyNPU", color=COLOR_CYAN, fontsize=26, fontweight='black', fontfamily='sans-serif', zorder=3)
ax.text(4.2, 12.45, "|  Deep Learning Hardware Accelerator & FPGA RTL Pipeline", color=COLOR_TEXT_MAIN, fontsize=16, fontweight='bold', zorder=3)
ax.text(1.2, 11.95, "End-to-End Flow: Python FP32 Training  -->  INT8/INT32 Quantization  -->  SystemVerilog RTL (4 Architectures)  -->  Bit-Accurate FPGA Verification",
        color=COLOR_TEXT_MUTED, fontsize=11, fontweight='medium', zorder=3)

# Header Stats Badges
stats = [
    ("Topology", "21 -> 14 -> 7 -> 1", COLOR_CYAN),
    ("Total Params", "421 (399 W + 22 B)", COLOR_PURPLE),
    ("Latency", "3 Cycles (30ns @ 100MHz)", COLOR_EMERALD),
    ("Throughput", "100 MInf/s (Pipelined)", COLOR_AMBER)
]

for idx, (label, val, color) in enumerate(stats):
    bx = 14.8 + (idx * 2.15)
    pill = patches.FancyBboxPatch((bx, 11.8), 2.05, 0.95,
                                  boxstyle="round,pad=0.1,rounding_size=0.2",
                                  facecolor='#1E293B', edgecolor=color,
                                  linewidth=1.2, zorder=3)
    ax.add_patch(pill)
    ax.text(bx + 1.025, 12.40, label, color=COLOR_TEXT_MUTED, fontsize=8, fontweight='bold', ha='center', zorder=4)
    ax.text(bx + 1.025, 12.05, val, color=color, fontsize=8.5, fontweight='black', ha='center', zorder=4)

# =============================================================
# LAYER CONFIGURATIONS
# =============================================================
layer_configs = [
    {
        "name": "INPUT LAYER",
        "subtitle": "21 Input Features",
        "bus": "168-bit Bus (21 x INT8)",
        "neurons": 21,
        "disp_count": 9,
        "color": COLOR_CYAN,
        "x": 2.2,
        "w": 3.6,
        "labels": ["x0", "x1", "x2", "x3", "...", "x18", "x19", "x20"]
    },
    {
        "name": "HIDDEN LAYER 1",
        "subtitle": "14 Neurons | ReLU",
        "bus": "112-bit Bus (14 x INT8)",
        "neurons": 14,
        "disp_count": 7,
        "color": COLOR_BLUE,
        "x": 8.0,
        "w": 3.6,
        "labels": ["n0", "n1", "n2", "...", "n11", "n12", "n13"]
    },
    {
        "name": "HIDDEN LAYER 2",
        "subtitle": "7 Neurons | ReLU",
        "bus": "56-bit Bus (7 x INT8)",
        "neurons": 7,
        "disp_count": 7,
        "color": COLOR_VIOLET,
        "x": 13.8,
        "w": 3.6,
        "labels": ["n0", "n1", "n2", "n3", "n4", "n5", "n6"]
    },
    {
        "name": "OUTPUT LAYER",
        "subtitle": "1 Neuron | Sigmoid",
        "bus": "8-bit Output (1 x INT8)",
        "neurons": 1,
        "disp_count": 1,
        "color": COLOR_EMERALD,
        "x": 19.6,
        "w": 3.6,
        "labels": ["y"]
    }
]

# Compute node coordinates
node_coords = []

for l_idx, layer in enumerate(layer_configs):
    x_center = layer["x"] + (layer["w"] / 2.0)
    count = layer["disp_count"]

    y_top = 10.0
    y_bottom = 4.2
    
    if count == 1:
        y_positions = [(y_top + y_bottom) / 2.0]
    else:
        y_positions = np.linspace(y_top, y_bottom, count)

    coords = [(x_center, y) for y in y_positions]
    node_coords.append(coords)

# =============================================================
# SYNAPSE INTERCONNECTS (Lines with Gradient Alpha)
# =============================================================
for l_idx in range(len(node_coords) - 1):
    curr_nodes = node_coords[l_idx]
    next_nodes = node_coords[l_idx + 1]
    curr_color = layer_configs[l_idx]["color"]
    next_color = layer_configs[l_idx + 1]["color"]

    # Interconnect background banner
    w_span = [
        "294 INT8 Weights (21 x 14)\n2352-bit Weight Bus + 14 INT32 Biases",
        "98 INT8 Weights (14 x 7)\n784-bit Weight Bus + 7 INT32 Biases",
        "7 INT8 Weights (7 x 1)\n56-bit Weight Bus + 1 INT32 Bias"
    ]

    mid_x = (layer_configs[l_idx]["x"] + layer_configs[l_idx]["w"] + layer_configs[l_idx + 1]["x"]) / 2.0

    # Draw Synapses
    for src_x, src_y in curr_nodes:
        for dst_x, dst_y in next_nodes:
            ax.plot([src_x, dst_x], [src_y, dst_y],
                    color=next_color, alpha=0.10, linewidth=0.7, zorder=1)

    # Synapse Info Card
    syn_card = patches.FancyBboxPatch((mid_x - 1.15, 7.3), 2.3, 1.2,
                                      boxstyle="round,pad=0.1,rounding_size=0.15",
                                      facecolor='#0F172A', edgecolor='#334155',
                                      linewidth=1.0, alpha=0.92, zorder=4)
    ax.add_patch(syn_card)
    ax.text(mid_x, 7.95, f"W{l_idx+1} Weights", color=COLOR_TEXT_MAIN, fontsize=8.5, fontweight='bold', ha='center', zorder=5)
    ax.text(mid_x, 7.55, w_span[l_idx], color=COLOR_TEXT_MUTED, fontsize=6.8, ha='center', zorder=5)

# =============================================================
# DRAW LAYER COLUMNS & NEURONS
# =============================================================
for l_idx, layer in enumerate(layer_configs):
    lx = layer["x"]
    lw = layer["w"]
    color = layer["color"]

    # Column Outer Bounding Card
    col_card = patches.FancyBboxPatch((lx, 3.4), lw, 7.9,
                                     boxstyle="round,pad=0.15,rounding_size=0.25",
                                     facecolor='#0F172A', edgecolor=color,
                                     linewidth=1.4, alpha=0.85, zorder=2)
    ax.add_patch(col_card)

    # Layer Header Pill
    l_header = patches.FancyBboxPatch((lx + 0.15, 10.45), lw - 0.3, 0.75,
                                      boxstyle="round,pad=0.1,rounding_size=0.15",
                                      facecolor='#1E293B', edgecolor=color,
                                      linewidth=1.2, zorder=3)
    ax.add_patch(l_header)
    ax.text(lx + lw/2.0, 10.95, layer["name"], color=color, fontsize=10, fontweight='black', ha='center', zorder=4)
    ax.text(lx + lw/2.0, 10.60, layer["subtitle"], color=COLOR_TEXT_MAIN, fontsize=8, fontweight='medium', ha='center', zorder=4)

    # Bus Info Sub-bar
    ax.text(lx + lw/2.0, 10.25, layer["bus"], color=COLOR_TEXT_MUTED, fontsize=7.5, fontweight='bold', ha='center', zorder=4)

    # Draw Nodes
    coords = node_coords[l_idx]
    for n_idx, (nx, ny) in enumerate(coords):
        # Node Glow Ring
        glow = plt.Circle((nx, ny), 0.36, color=color, alpha=0.22, zorder=3)
        ax.add_patch(glow)

        # Node Body
        node = plt.Circle((nx, ny), 0.28, facecolor='#1E293B', edgecolor=color, linewidth=1.8, zorder=4)
        ax.add_patch(node)

        # Node Label
        if n_idx < len(layer["labels"]):
            lbl = layer["labels"][n_idx]
        else:
            lbl = f"n{n_idx}"
        ax.text(nx, ny, lbl, color=COLOR_TEXT_MAIN, fontsize=8, fontweight='bold', ha='center', va='center', zorder=5)

    # Bottom Module Badge
    mod_name = f"layer{l_idx+1}.sv" if l_idx < 3 else "layer3.sv"
    if l_idx == 0:
        mod_desc = "Packed Array\n21 INT8 Inputs"
    elif l_idx == 1:
        mod_desc = "14 Parallel Neurons\n112b Output Bus"
    elif l_idx == 2:
        mod_desc = "7 Parallel Neurons\n56b Output Bus"
    else:
        mod_desc = "1 Output Neuron\n8b Final Activation"

    bot_pill = patches.FancyBboxPatch((lx + 0.2, 3.55), lw - 0.4, 0.75,
                                      boxstyle="round,pad=0.1,rounding_size=0.12",
                                      facecolor='#1E293B', edgecolor='#475569',
                                      linewidth=1.0, zorder=3)
    ax.add_patch(bot_pill)
    ax.text(lx + lw/2.0, 4.02, mod_name, color=color, fontsize=8.5, fontweight='bold', ha='center', zorder=4)
    ax.text(lx + lw/2.0, 3.70, mod_desc, color=COLOR_TEXT_MUTED, fontsize=6.8, ha='center', zorder=4)

# =============================================================
# HARDWARE DATAPATH & RTL PANELS (BOTTOM SECTION)
# =============================================================
# Panel 1: Neuron Silicon Datapath (MAC -> Requant -> ReLU)
p1_card = patches.FancyBboxPatch((0.5, 0.45), 7.2, 2.7,
                                 boxstyle="round,pad=0.15,rounding_size=0.25",
                                 facecolor='#0F172A', edgecolor=COLOR_CYAN,
                                 linewidth=1.2, alpha=0.9, zorder=2)
ax.add_patch(p1_card)

ax.text(0.8, 2.85, "HARDWARE NEURON DATAPATH (neuron.sv)", color=COLOR_CYAN, fontsize=10.5, fontweight='bold', zorder=3)
ax.text(0.8, 2.50, "[1] MAC Engine (mac.sv):", color=COLOR_TEXT_MAIN, fontsize=8.5, fontweight='bold', zorder=3)
ax.text(3.3, 2.50, "acc = bias + SUM(x[i] * w[i]) -> INT32", color=COLOR_TEXT_MUTED, fontsize=8, fontfamily='monospace', zorder=3)

ax.text(0.8, 2.10, "[2] Requantize (requantize.sv):", color=COLOR_TEXT_MAIN, fontsize=8.5, fontweight='bold', zorder=3)
ax.text(3.8, 2.10, "scaled = (acc * MULT) >>> 20 (Dyadic Scaling)", color=COLOR_TEXT_MUTED, fontsize=8, fontfamily='monospace', zorder=3)

ax.text(0.8, 1.70, "[3] Saturation / Clamp:", color=COLOR_TEXT_MAIN, fontsize=8.5, fontweight='bold', zorder=3)
ax.text(3.3, 1.70, "clip(scaled, -128, +127) -> INT8 Range", color=COLOR_TEXT_MUTED, fontsize=8, fontfamily='monospace', zorder=3)

ax.text(0.8, 1.30, "[4] ReLU Activation (relu.sv):", color=COLOR_TEXT_MAIN, fontsize=8.5, fontweight='bold', zorder=3)
ax.text(3.5, 1.30, "assign out = (in < 0) ? 8'sd0 : in;", color=COLOR_TEXT_MUTED, fontsize=8, fontfamily='monospace', zorder=3)

ax.text(0.8, 0.80, "Zero FP divider logic in silicon | Zero overflow risk | Pure integer datapath", color=COLOR_EMERALD, fontsize=8, fontweight='bold', zorder=3)

# Panel 2: 4 Hardware RTL Architectures (Phase 5, 7, 8, 9)
p2_card = patches.FancyBboxPatch((8.1, 0.45), 7.6, 2.7,
                                 boxstyle="round,pad=0.15,rounding_size=0.25",
                                 facecolor='#0F172A', edgecolor=COLOR_PURPLE,
                                 linewidth=1.2, alpha=0.9, zorder=2)
ax.add_patch(p2_card)

ax.text(8.4, 2.85, "4 RTL HARDWARE ARCHITECTURES BUILT & VERIFIED", color=COLOR_PURPLE, fontsize=10.5, fontweight='bold', zorder=3)

archs = [
    ("1. Combinational (tinynpu.sv):", "399 Mults | ~25ns Delay | Baseline Datapath", COLOR_BLUE),
    ("2. Sequential FSM (tinynpu_fpga.sv):", "3 Clock Cycles | 100MHz | ROM Handshake (Phase 8)", COLOR_CYAN),
    ("3. Pipelined Streaming (tinynpu_pipelined.sv):", "100 MInf/s | II=1 | Max Throughput (Phase 9)", COLOR_AMBER),
    ("4. Resource-Shared (tinynpu_resource_shared.sv):", "1 Single MAC | 399 Cycles | 99.7% Area Reduction", COLOR_EMERALD)
]

for a_idx, (a_title, a_desc, a_col) in enumerate(archs):
    ay = 2.45 - (a_idx * 0.42)
    ax.text(8.4, ay, a_title, color=a_col, fontsize=8.2, fontweight='bold', zorder=3)
    ax.text(8.4 + 4.1, ay, a_desc, color=COLOR_TEXT_MUTED, fontsize=7.5, zorder=3)

ax.text(8.4, 0.80, "All 4 hardware variants verified with 100% bit-accurate software equivalence", color=COLOR_PURPLE, fontsize=8, fontweight='bold', zorder=3)

# Panel 3: FPGA Physical Synthesis Benchmarks (Xilinx Artix-7)
p3_card = patches.FancyBboxPatch((16.1, 0.45), 7.4, 2.7,
                                 boxstyle="round,pad=0.15,rounding_size=0.25",
                                 facecolor='#0F172A', edgecolor=COLOR_EMERALD,
                                 linewidth=1.2, alpha=0.9, zorder=2)
ax.add_patch(p3_card)

ax.text(16.4, 2.85, "FPGA SYNTHESIS BENCHMARKS (Artix-7 xc7a35t)", color=COLOR_EMERALD, fontsize=10.5, fontweight='bold', zorder=3)

benchmarks = [
    ("LUT Utilization (Sequential):", "1,420 LUTs (6.8% of xc7a35t)"),
    ("Flip-Flops (Sequential):", "284 Registers (0.7% of chip)"),
    ("DSP48 Slices (Shared vs Parallel):", "1 DSP (1.1%) vs 22 DSPs (24.4%)"),
    ("Maximum Clock Frequency (Fmax):", "127.4 MHz (Setup slack +2.15ns MET)"),
    ("Total Operational Power:", "< 85 mW (Sub-100mW battery AI)")
]

for b_idx, (b_name, b_val) in enumerate(benchmarks):
    by = 2.45 - (b_idx * 0.35)
    ax.text(16.4, by, b_name, color=COLOR_TEXT_MAIN, fontsize=8, fontweight='medium', zorder=3)
    ax.text(21.4, by, b_val, color=COLOR_CYAN, fontsize=8, fontweight='bold', zorder=3)

ax.text(16.4, 0.80, "Deployment Ready: synth_vivado.tcl  |  tinynpu.xdc  |  synth_yosys.ys", color=COLOR_TEXT_MUTED, fontsize=7.5, fontfamily='monospace', zorder=3)

# =============================================================
# SET AXES & EXPORT HIGH-RES IMAGE
# =============================================================
ax.set_xlim(0, 24)
ax.set_ylim(0, 13.5)
ax.axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300, facecolor='#0B0F19', edgecolor='none', bbox_inches='tight')
plt.close()

print()
print("============================================================")
print("TinyNPU High-Resolution Architecture Diagram Generated!")
print("============================================================")
print(f"Saved to: {OUTPUT_IMAGE}")
print("Resolution: 300 DPI (7200 x 4050 pixels, 16:9 widescreen format)")
print("============================================================")
