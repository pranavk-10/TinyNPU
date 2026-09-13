# Quick Reference Guide

## Project Structure at a Glance

```
TinyNPU/
├── 📘 README.md ..................... Main documentation
├── 📘 PROJECT_STRUCTURE.md .......... Detailed file organization guide
├── 📄 requirements.txt .............. Python dependencies
│
├── 📂 src/ (Python ML Pipeline)
│   ├── model.py .................... FP32 neural network definition
│   ├── backprop.py ................. Training algorithm
│   ├── load.py ..................... Data loading
│   ├── quantize_model.py ........... Quantization (FP32 → INT8/INT32)
│   └── quantized_inference.py ...... Run inference on quantized model
│
├── 📂 tools/ (Executables & Utils)
│   ├── main.py ..................... ⭐ RUN THIS FIRST - Main pipeline
│   ├── export_mem.py ............... Export to .mem format for RTL
│   └── create_architecture_image.py  Generate diagrams
│
├── 📂 hardware/ (SystemVerilog RTL)
│   └── rtl/
│       ├── components/ ............ Atomic units (MAC, ReLU, Requantize)
│       ├── layers/ ............... Layer implementations (14, 7, 1 neuron)
│       ├── top/ .................. Main network module
│       ├── testbenches/ .......... Test files (.sv)
│       └── sim/ .................. Simulation outputs (gitignored)
│
├── 📂 data/ (Datasets & Parameters)
│   ├── inputs/ ................... Dataset (CSV)
│   ├── quantized/ ................ Scale factors (TXT)
│   └── mem/ ...................... RTL memory files (.mem)
│
└── 📂 docs/ (Documentation)
    └── tinynpu_architecture.png ... Architecture diagram
```

## Getting Started

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Full Pipeline
```bash
python tools/main.py
```

**Output:** Trains model, quantizes, exports .mem files and requantization header to `data/mem/`

### 3️⃣ Run Real-Model RTL Verification (Phase 7)
```bash
python tools/verify_rtl.py
```

**Output:** Verifies 100% bit-accurate parity between Python reference and RTL hardware datapath!

### 4️⃣ Simulate Hardware with Icarus Verilog (Optional)
```bash
# Using Makefile
cd hardware
make real_sim

# Or manually:
iverilog -g2012 -I.. -o sim/tinynpu_real_sim \
  rtl/components/*.sv rtl/layers/*.sv rtl/top/tinynpu.sv rtl/testbenches/tb_tinynpu_real.sv
vvp sim/tinynpu_real_sim
```

## Key Files Explained

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `src/model.py` | Defines the network architecture | - | FP32 network class |
| `src/backprop.py` | Training with backpropagation | CSV data | Trained FP32 weights |
| `src/quantize_model.py` | Converts FP32 → INT8/INT32 | FP32 weights | Quantized parameters |
| `tools/main.py` | **Main orchestrator** | CSV dataset | Trained & quantized model |
| `tools/export_mem.py` | Convert to .mem format for RTL | Quantized params | `.mem` files for hardware |
| `hardware/rtl/.../tinynpu.sv` | Complete network RTL | .mem files | RTL inference results |

## Network Architecture

```
Input: 21 features
   ↓ [294 weights]
Layer 1: 14 neurons (ReLU)
   ↓ [98 weights]
Layer 2: 7 neurons (ReLU)
   ↓ [7 weights]
Layer 3: 1 neuron (Sigmoid)
   ↓
Output: Binary prediction (0 or 1)
```

**Total Parameters:** 421 (399 weights + 22 biases)

## Data Flow

```
1. data/inputs/tinynpu_21_feature_dataset.csv
     ↓
2. tools/main.py → Python ML Pipeline
     ↓
3. FP32 Model (trained)
     ↓
4. Quantization → INT8/INT32
     ↓
5. data/quantized/ (scale factors)
   data/mem/ (.mem files)
     ↓
6. hardware/rtl/top/tinynpu.sv (RTL inference)
     ↓
7. RTL Simulation Results
```

## File Organization Summary

| Directory | Purpose | File Type | Gitignored? |
|-----------|---------|-----------|-------------|
| `src/` | ML algorithms | `.py` | No |
| `tools/` | Utilities & pipeline | `.py` | No |
| `hardware/rtl/components/` | Atomic RTL units | `.sv` | No |
| `hardware/rtl/layers/` | Layer implementations | `.sv` | No |
| `hardware/rtl/top/` | Main network | `.sv` | No |
| `hardware/rtl/testbenches/` | Test files | `.sv` | No |
| `hardware/sim/` | Simulation outputs | executable | **Yes** |
| `data/inputs/` | Raw dataset | `.csv` | No |
| `data/quantized/` | Scale factors | `.txt` | No |
| `data/mem/` | Memory files for RTL | `.mem`, `.txt` | No |
| `docs/` | Images & docs | `.png`, `.md` | No |

## Common Tasks

### Train the Model
```bash
python tools/main.py
```

### Export Only (No Training)
```python
from src.quantize_model import quantize_and_export
# See quantize_model.py for API
```

### Run Only Inference
```python
from src.quantized_inference import quantized_predict
# See quantized_inference.py for API
```

### Simulate One Component
```bash
cd hardware/rtl
iverilog -g2012 -o ../sim/mac_sim components/mac.sv testbenches/tb_mac.sv
../sim/mac_sim
```

### Simulate Full Network
```bash
cd hardware/rtl
iverilog -g2012 -o ../sim/tinynpu_sim \
  components/*.sv layers/*.sv top/tinynpu.sv testbenches/tb_tinynpu.sv
../sim/tinynpu_sim
```

## Directory Purposes

### 🟦 `src/` - Machine Learning
Core ML pipeline. Change here to:
- Modify network architecture
- Change training algorithm
- Adjust learning rate
- Implement new activations

### 🟩 `tools/` - Integration & Export
Pipeline orchestration. Change here to:
- Add new export formats
- Modify quantization parameters
- Add logging/visualization

### 🟪 `hardware/` - Digital Logic
RTL implementation. Change here to:
- Modify hardware datapath
- Add optimizations
- Change MAC strategy

### 🟨 `data/` - I/O
Model data and parameters. Change here to:
- Use different datasets
- Adjust quantization scales
- Add new test vectors

## Environment Setup

### Python
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Simulation Tools
```bash
# Install Icarus Verilog (Windows with iverilog in PATH)
# Or set path to iverilog explicitly
```

## Files to Never Modify
- `.gitignore` - Except to add new exclusions
- `requirements.txt` - Update only when adding dependencies
- **Auto-generated files in `hardware/sim/`** - These are simulation outputs

## Next Steps

1. **Read** `README.md` for full project overview
2. **Read** `PROJECT_STRUCTURE.md` for detailed file organization
3. **Run** `python tools/main.py` to train and quantize
4. **Explore** `hardware/rtl/` for RTL implementation
5. **Simulate** components individually before running full simulation

---

**Need help?** See README.md for detailed documentation.
