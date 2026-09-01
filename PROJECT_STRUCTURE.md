# TinyNPU Project Structure

This document describes the organization of the TinyNPU repository.

## Directory Organization

```
TinyNPU/
│
├── README.md                    # Main project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── src/                         # Python source code (ML pipeline)
│   ├── model.py                 # FP32 neural network definition
│   ├── backprop.py              # Backpropagation training algorithm
│   ├── load.py                  # Dataset loading utilities
│   ├── quantize_model.py        # INT8/INT32 quantization pipeline
│   └── quantized_inference.py   # Inference on quantized model
│
├── tools/                       # Utility and orchestration scripts
│   ├── main.py                  # Main pipeline runner (entry point)
│   ├── export_mem.py            # Export quantized params to .mem format
│   └── create_architecture_image.py # Generate architecture diagrams
│
├── hardware/                    # SystemVerilog RTL implementation
│   │
│   ├── rtl/                     # RTL source code
│   │   ├── components/          # Atomic hardware modules
│   │   │   ├── mac.sv           # Multiply-Accumulate unit (INT8×INT8→INT32)
│   │   │   ├── relu.sv          # ReLU activation (max(0, x))
│   │   │   └── requantize.sv    # Requantization (INT32→INT8)
│   │   │
│   │   ├── layers/              # Layer implementations
│   │   │   ├── neuron.sv        # Single neuron (MAC → Requantize → ReLU)
│   │   │   ├── layer1.sv        # 14-neuron layer (21 → 14)
│   │   │   ├── layer2.sv        # 7-neuron layer (14 → 7)
│   │   │   └── layer3.sv        # 1-neuron layer (7 → 1)
│   │   │
│   │   ├── top/                 # Top-level module
│   │   │   └── tinynpu.sv       # Complete TinyNPU network (21→14→7→1)
│   │   │
│   │   └── testbenches/         # Simulation testbenches
│   │       ├── tb_mac.sv        # Test MAC unit
│   │       ├── tb_relu.sv       # Test ReLU activation
│   │       ├── tb_requantize.sv # Test requantization
│   │       ├── tb_neuron.sv     # Test single neuron
│   │       ├── tb_layer1.sv     # Test layer 1
│   │       ├── tb_layer2.sv     # Test layer 2
│   │       ├── tb_layer3.sv     # Test layer 3
│   │       └── tb_tinynpu.sv    # Test complete network
│   │
│   └── sim/                     # Simulation outputs (generated, in .gitignore)
│       ├── tinynpu_sim          # Main network simulation
│       ├── layer1_sim           # Layer 1 simulation
│       ├── layer2_sim           # Layer 2 simulation
│       ├── layer3_sim           # Layer 3 simulation
│       ├── neuron_sim           # Single neuron simulation
│       ├── mac_sim              # MAC unit simulation
│       ├── relu_sim             # ReLU activation simulation
│       └── requantize_sim       # Requantization simulation
│
├── data/                        # Datasets and model parameters
│   │
│   ├── inputs/                  # Input datasets
│   │   └── tinynpu_21_feature_dataset.csv  # Training data (21 features, binary classification)
│   │
│   ├── quantized/               # Quantized model parameters (scale factors)
│   │   ├── input_scale.txt
│   │   ├── weights1_scale.txt
│   │   ├── weights2_scale.txt
│   │   ├── weights3_scale.txt
│   │   ├── activation1_scale.txt
│   │   ├── activation2_scale.txt
│   │   ├── bias1_scale.txt
│   │   ├── bias2_scale.txt
│   │   └── bias3_scale.txt
│   │
│   └── mem/                     # Hardware memory files (.mem format)
│       ├── scales.txt           # Quantization scale factors for RTL
│       ├── inputs.mem           # INT8 input values
│       ├── weights1.mem         # INT8 weights (layer 1)
│       ├── bias1.mem            # INT32 biases (layer 1)
│       ├── weights2.mem         # INT8 weights (layer 2)
│       ├── bias2.mem            # INT32 biases (layer 2)
│       ├── weights3.mem         # INT8 weights (layer 3)
│       └── bias3.mem            # INT32 biases (layer 3)
│
└── docs/                        # Documentation and diagrams
    └── tinynpu_architecture.png # Network architecture diagram
```

## File Organization Principles

### `src/` - Machine Learning Pipeline
Core Python modules for:
- Neural network definition (FP32)
- Training with backpropagation
- Data loading
- Quantization (FP32 → INT8/INT32)
- Quantized inference

**Use when:** Developing ML algorithms, training, or quantization logic

### `tools/` - Utilities & Orchestration
Executable scripts:
- `main.py` - Runs the complete ML→RTL pipeline
- `export_mem.py` - Converts quantized parameters to RTL-compatible .mem format
- `create_architecture_image.py` - Generates documentation diagrams

**Use when:** Running the pipeline or exporting data for hardware

### `hardware/` - SystemVerilog RTL
Organized by functionality:
- **components/** - Atomic building blocks (MAC, ReLU, requantize)
- **layers/** - Layer implementations using components
- **top/** - Complete network hierarchy
- **testbenches/** - Simulation and verification
- **sim/** - Generated simulation outputs (gitignored)

**Use when:** Implementing, modifying, or simulating hardware

### `data/` - Datasets & Parameters
- **inputs/** - Original datasets
- **quantized/** - Intermediate quantization metadata
- **mem/** - Hardware-ready memory format for RTL

**Use when:** Training new models or deploying to hardware

### `docs/` - Documentation
Project documentation, diagrams, and images.

## Data Flow

```
data/inputs/tinynpu_21_feature_dataset.csv
           ↓
       src/load.py
           ↓
       src/model.py (FP32 training)
           ↓
       src/backprop.py
           ↓
       Trained FP32 Model
           ↓
       src/quantize_model.py
           ↓
       data/quantized/ (scale factors)
           ↓
       tools/export_mem.py
           ↓
       data/mem/ (.mem files)
           ↓
       hardware/rtl/top/tinynpu.sv
           ↓
       hardware/rtl/testbenches/tb_tinynpu.sv
           ↓
       RTL Simulation Results
```

## Running the Project

### 1. Train and Quantize
```bash
# From root directory
python tools/main.py
```

This will:
- Load dataset from `data/inputs/`
- Train FP32 model
- Quantize to INT8/INT32
- Export to `data/mem/` as .mem files
- Generate output in `data/quantized/`

### 2. Simulate Hardware
```bash
# Simulate MAC unit
cd hardware/rtl
iverilog -g2012 -o ../sim/mac_sim components/mac.sv testbenches/tb_mac.sv
../sim/mac_sim

# Simulate complete network
iverilog -g2012 -o ../sim/tinynpu_sim \
  components/*.sv \
  layers/*.sv \
  top/tinynpu.sv \
  testbenches/tb_tinynpu.sv
../sim/tinynpu_sim
```

## Adding New Components

### Adding a New RTL Module
1. Create `.sv` file in appropriate subdirectory:
   - `hardware/rtl/components/` - For atomic units
   - `hardware/rtl/layers/` - For layer implementations
2. Create corresponding testbench in `hardware/rtl/testbenches/`
3. Update README with new module description

### Adding Training Features
1. Add logic to `src/` modules
2. Update `tools/main.py` to include new steps
3. Document in README

## Version Control

**Included in Git:**
- All source code (src/, hardware/rtl/)
- Documentation (README.md, docs/)
- Configuration (.gitignore, requirements.txt)
- Dataset (data/inputs/*.csv)

**Excluded from Git** (in .gitignore):
- Generated simulation outputs (hardware/sim/)
- Model artifacts (*.npy files)
- Memory files (*.mem)
- Python cache (__pycache__/)
- Generated scale factors

To preserve quantized parameters across commits, explicitly add to Git:
```bash
git add -f data/quantized/
git add -f data/mem/
```

## Repository Statistics

| Category | Count |
|----------|-------|
| Python modules | 5 |
| Utility scripts | 3 |
| RTL components | 3 |
| Layer modules | 4 |
| Testbenches | 8 |
| Total RTL files | 16 |
| Total trainable parameters | 421 |
| Network architecture | 21→14→7→1 |

