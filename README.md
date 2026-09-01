# TinyNPU

**TinyNPU** is an end-to-end neural-network inference accelerator project that demonstrates the complete pipeline from **Python-based ML training to hardware implementation in SystemVerilog RTL**.

The project showcases how a trained neural network model is progressively optimized, quantized, and implemented as a specialized hardware inference accelerator for FPGA deployment.

## Network Architecture

```
Input (21 features)
    ↓ [21 × 14 weights]
Hidden Layer 1 (14 neurons, ReLU)
    ↓ [14 × 7 weights]
Hidden Layer 2 (7 neurons, ReLU)
    ↓ [7 × 1 weights]
Output Layer (1 neuron, Sigmoid)
    ↓
Binary Classification
```

## Complete Pipeline

```
CSV Dataset
    ↓
FP32 Training (Backpropagation)
    ↓
Save Weights + Biases (FP32)
    ↓
FP32 Inference & Validation
    ↓
INT8 / INT32 Quantization
    ↓
Quantized Inference
    ↓
Export to .mem (RTL-ready)
    ↓
SystemVerilog RTL Implementation
    ↓
RTL Simulation & Verification
    ↓
FPGA Deployment
```

---

## Directory Structure

```
TinyNPU/
├── main.py                          # Main pipeline orchestrator
├── model.py                         # Neural network class (FP32)
├── backprop.py                      # Training with backpropagation
├── load.py                          # Dataset loading
├── quantize_model.py                # INT8/INT32 quantization
├── quantized_inference.py           # Inference on quantized model
├── export_mem.py                    # Export to .mem format
├── create_architecture_image.py     # Generates architecture diagram
├── tinynpu_21_feature_dataset.csv   # Training dataset (21 features)
├── requirements.txt                 # Dependencies
├── docs/                            # Documentation
│   └── tinynpu_architecture.png
├── rtl/                             # SystemVerilog RTL modules
├── mem/                             # Exported .mem files (weights/biases)
├── quantized/                       # Quantized parameters
├── tinynpu_sim/                     # Simulation outputs
├── layer1_sim/                      # Layer 1 simulation
├── layer2_sim/                      # Layer 2 simulation
├── layer3_sim/                      # Layer 3 simulation
├── mac_sim/                         # MAC unit simulation
├── neuron_sim/                      # Neuron simulation
└── relu_sim/                        # ReLU activation simulation
```

## Getting Started

### Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.26
- pandas >= 2.0
- matplotlib >= 3.8

### Running the Pipeline

Execute the complete pipeline:

```bash
python main.py
```

This will:
1. Load the dataset from CSV
2. Create and train the neural network (FP32)
3. Save trained weights and biases
4. Run FP32 inference
5. Quantize to INT8/INT32
6. Export parameters for hardware implementation

## Training Details

### Algorithm

- **Optimizer:** Gradient Descent
- **Loss Function:** Binary Cross-Entropy
- **Learning Rate:** 0.01 (configurable)
- **Epochs:** 1000 (configurable)

### Activation Functions

- **Hidden Layers:** ReLU (Rectified Linear Unit)
- **Output Layer:** Sigmoid (Binary classification)

### Backpropagation

The `backprop.py` module implements standard backpropagation:

1. Forward pass through all layers
2. Compute loss via Binary Cross-Entropy
3. Backward pass to compute gradients
4. Update weights and biases via gradient descent

## Quantization Strategy

### INT8 Quantization

Weights, inputs, and activations are converted to INT8 (-128 to 127):

```
FP32 Value → Quantize (divide by scale) → INT8 Value
INT8 Value → Dequantize (multiply by scale) → FP32 Value
```

### INT32 for Accumulators

Biases and MAC accumulations use INT32 for numerical precision:

```
INT8 × INT8 → INT32 (MAC output)
INT32 + INT32 (Bias) → INT32 (Pre-activation)
INT32 → Requantize → INT8 (Activation)
```

## Hardware Export

### .mem Files

The `export_mem.py` script exports quantized parameters in `.mem` format for RTL:

- `weights_layer1.mem` – INT8 weights
- `bias_layer1.mem` – INT32 biases
- `weights_layer2.mem`
- `bias_layer2.mem`
- `weights_layer3.mem`
- `bias_layer3.mem`
- `scale_factors.mem` – Quantization scales

### RTL Implementation

SystemVerilog modules in the `rtl/` directory implement:

- MAC (Multiply-Accumulate) units
- Quantization/Dequantization logic
- ReLU activation
- Sigmoid activation
- Layer pipelines

## Verification

The project includes simulation modules to verify hardware against Python:

- `tinynpu_sim/` – Full-system simulation
- `mac_sim/` – MAC unit simulation
- `relu_sim/` – ReLU activation simulation
- `neuron_sim/` – Single neuron simulation

## Quantized Inference

The Python implementation also performs inference using the quantized representation.

The two implementations are compared:

```text
FP32 Model
     │
     └──────→ FP32 Probability

Quantized Model
     │
     └──────→ INT8 Probability
```

The quantized inference results are very close to the FP32 reference on the current test samples.

The current quantized test achieved:

```text
Correct: 2/2
Accuracy: 100.00%
```

This accuracy refers to the current tiny test set and should not be interpreted as meaningful real-world model accuracy.

---

## Hardware Datapath

Each neural-network neuron follows the basic hardware datapath:

```text
              INT8 Input
                   │
                   ▼
            ┌─────────────┐
            │ INT8 × INT8 │
            │     MAC     │
            └──────┬──────┘
                   │
                   ▼
            INT32 Accumulator
                   │
                   +
               INT32 Bias
                   │
                   ▼
            Requantization
                   │
                   ▼
                 INT8
                   │
                   ▼
                 ReLU
                   │
                   ▼
              INT8 Output
```

The MAC operation is conceptually:

```text
accumulator = bias

for each input:

    accumulator += input × weight
```

The INT32 accumulator is then converted back into an INT8 activation through fixed-point requantization.

---

## RTL Architecture

The hardware inference implementation is written in **SystemVerilog**.

The neural network is implemented as:

```text
21 → 14 → 7 → 1
```

The RTL hierarchy is:

```text
                    TinyNPU
                       │
                       ▼
                ┌─────────────┐
                │   Layer 1   │
                │   21 → 14   │
                └──────┬──────┘
                       │
                 14 × INT8
                       │
                       ▼
                ┌─────────────┐
                │   Layer 2   │
                │    14 → 7   │
                └──────┬──────┘
                       │
                  7 × INT8
                       │
                       ▼
                ┌─────────────┐
                │   Layer 3   │
                │     7 → 1   │
                └──────┬──────┘
                       │
                       ▼
                  1 × INT8
```

---

## Core Python Modules

| Module | Purpose |
|--------|---------|
| `model.py` | FP32 neural network class |
| `backprop.py` | Backpropagation training algorithm |
| `load.py` | CSV dataset loading |
| `quantize_model.py` | INT8/INT32 quantization pipeline |
| `quantized_inference.py` | Inference on quantized model |
| `export_mem.py` | Export to .mem format for RTL |
| `create_architecture_image.py` | Generate architecture diagrams |

## Parameter Summary

### Total Trainable Parameters: **421**

| Layer | Shape | Weights | Biases |
|-------|------:|--------:|-------:|
| Layer 1 (21→14) | (14, 21) | 294 | 14 |
| Layer 2 (14→7) | (7, 14) | 98 | 7 |
| Layer 3 (7→1) | (1, 7) | 7 | 1 |
| **Total** | - | **399** | **22** |

## RTL Hardware Implementation

### Bus Widths

| Layer | Output Nodes | Total Bus Width |
|-------|-------------:|----------------:|
| Layer 1 Output | 14 × INT8 | 112 bits |
| Layer 2 Output | 7 × INT8 | 56 bits |
| Layer 3 Output | 1 × INT8 | 8 bits |

### RTL Modules

```
rtl/
├── Computation Units
│   ├── mac.sv               # Multiply-Accumulate
│   ├── relu.sv              # ReLU activation
│   ├── requantize.sv        # INT32 → INT8 conversion
│   └── neuron.sv            # Complete neuron pipeline
│
├── Layer Implementations
│   ├── layer1.sv            # 14 neurons (21→14)
│   ├── layer2.sv            # 7 neurons (14→7)
│   ├── layer3.sv            # 1 neuron (7→1)
│   └── tinynpu.sv           # Complete network integration
│
└── Testbenches
    ├── tb_mac.sv
    ├── tb_relu.sv
    ├── tb_requantize.sv
    ├── tb_neuron.sv
    ├── tb_layer1.sv
    ├── tb_layer2.sv
    ├── tb_layer3.sv
    └── tb_tinynpu.sv
```

### Verified Components

- ✓ MAC Unit – INT8 × INT8 → INT32 accumulation
- ✓ ReLU – max(0, x) activation
- ✓ Requantization – INT32 → INT8 with saturation
- ✓ Single Neuron – Complete datapath
- ✓ Layer 1 – All 14 neurons verified
- ✓ Layer 2 – All 7 neurons verified
- ✓ Layer 3 – Output neuron verified
- ✓ Complete Network – Full integration test passed

## Hardware Memory Files

The `mem/` directory contains exported quantized parameters:

```
mem/
├── inputs.mem           # INT8 input features
├── weights1.mem         # INT8 weights (Layer 1)
├── bias1.mem            # INT32 biases (Layer 1)
├── weights2.mem         # INT8 weights (Layer 2)
├── bias2.mem            # INT32 biases (Layer 2)
├── weights3.mem         # INT8 weights (Layer 3)
├── bias3.mem            # INT32 biases (Layer 3)
└── scales.txt           # Quantization scale factors
```

## Dataset

- **File:** `tinynpu_21_feature_dataset.csv`
- **Input Features:** 21
- **Target:** Binary classification (0 or 1)
- **Task:** Train a compact neural network on tabular data

## Example Workflow

### 1. Train the Model

```bash
python main.py
```

Output:
```
Dataset loaded
Input shape: (n_samples, 21)
Target shape: (n_samples,)

Neural Network Architecture:
Input Layer  : 21
Hidden Layer : 14
Hidden Layer : 7
Output Layer : 1

Starting Training...
[Training Progress...]
FP32 weights and biases exported!
```

### 2. FP32 Inference

The trained model generates predictions:

```
Sample 1: Target = 0, Probability = 0.1234, Prediction = 0
Sample 2: Target = 1, Probability = 0.8567, Prediction = 1
```

### 3. Quantization

The model is automatically quantized to INT8/INT32:

```
Quantizing weights...
Quantizing biases...
Quantizing activations...
Complete!
```

### 4. Export for Hardware

Parameters are saved to `.mem` files:

```
weights1.mem    → 294 INT8 values
bias1.mem       → 14 INT32 values
weights2.mem    → 98 INT8 values
bias2.mem       → 7 INT32 values
weights3.mem    → 7 INT8 values
bias3.mem       → 1 INT32 value
```

### 5. RTL Simulation

SystemVerilog testbenches verify the hardware:

```bash
# Simulate individual components
iverilog -o tb_mac tb_mac.sv mac.sv
./tb_mac

# Simulate complete network
iverilog -o tb_tinynpu tb_tinynpu.sv tinynpu.sv layer1.sv layer2.sv layer3.sv ...
./tb_tinynpu
```

## Project Goal

**Demonstrate how a trained ML model can be efficiently implemented in hardware:**

✓ Software ML pipeline (training, quantization)  
✓ Fixed-point integer arithmetic  
✓ Hardware description (SystemVerilog RTL)  
✓ Simulation & verification  
✓ FPGA deployment readiness  

## Learning Outcomes

This project teaches:

- Neural network fundamentals (forward/backward propagation)
- Quantization techniques for hardware implementation
- SystemVerilog hardware design
- Testing and verification of hardware modules
- The ML→Hardware translation pipeline

---

**Repository:** [pranavk-10/TinyNPU](https://github.com/pranavk-10/TinyNPU)ct requirements.

---

## Python Environment

The project uses Python with NumPy and Pandas.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Python pipeline:

```bash
python main.py
```

The Python pipeline performs:

```text
Dataset Loading
      ↓
FP32 Training
      ↓
FP32 Inference
      ↓
Full Quantization
      ↓
Quantized Inference
      ↓
.mem Export
```

---

## Training Flow

The Python training pipeline follows:

```text
CSV
 ↓
Load Features + Targets
 ↓
Initialize Weights and Biases
 ↓
Forward Propagation
 ↓
Calculate Binary Cross-Entropy Loss
 ↓
Backpropagation
 ↓
Calculate Gradients
 ↓
Update Weights and Biases
 ↓
Repeat for Multiple Epochs
 ↓
Save Trained FP32 Parameters
```

The trained parameters are then passed into the quantization pipeline.

---

## Full Software-to-Hardware Flow

The complete project flow is:

```text
                  CSV DATASET
                       │
                       ▼
               Python Training
                       │
                       ▼
                 FP32 Model
                       │
                       ▼
               FP32 Inference
                       │
                       ▼
                 Quantization
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
       INT8 Weights          INT32 Biases
            │                     │
            └──────────┬──────────┘
                       │
                       ▼
              Quantized Inference
                       │
                       ▼
                  .mem Export
                       │
                       ▼
               SystemVerilog RTL
                       │
                       ▼
                 MAC Operations
                       │
                       ▼
               INT32 Accumulation
                       │
                       ▼
                 Requantization
                       │
                       ▼
                     ReLU
                       │
                       ▼
                  Next Layer
                       │
                       ▼
                  Final Output
                       │
                       ▼
                 RTL Prediction
```

---

## Verification Strategy

Python acts as the **golden reference model**.

The intended verification architecture is:

```text
                    Python
                    Reference
                       │
                       ▼
                Expected Output
                       │
                       │
                       ▼
                 ┌───────────┐
                 │ Comparator│
                 └─────┬─────┘
                       ▲
                       │
                       │
                  RTL Output
                       ▲
                       │
                  TinyNPU RTL
                       ▲
                       │
                    .mem
                    Files
```

The same quantized inputs and parameters should eventually be supplied to both implementations.

The verification goal is:

```text
Python Quantized Output
          =
RTL Hardware Output
```

subject to the same fixed-point arithmetic, scaling, saturation, and activation rules.

The individual RTL components and complete network structure have already been verified using deterministic test vectors.

The next major verification milestone is to run the RTL using the actual `.mem` files generated from the trained and quantized model.

---

## Current Verification Status

The current RTL verification has progressed hierarchically.

```text
MAC
 ↓
ReLU
 ↓
Requantization
 ↓
Neuron
 ↓
Layer 1
 ↓
Layer 2
 ↓
Layer 3
 ↓
Complete TinyNPU
```

All currently implemented RTL verification stages have passed.

| Component | Status |
|---|---|
| MAC | ✅ PASSED |
| ReLU | ✅ PASSED |
| Requantization | ✅ PASSED |
| Single Neuron | ✅ PASSED |
| Layer 1 — 21 → 14 | ✅ PASSED |
| Layer 2 — 14 → 7 | ✅ PASSED |
| Layer 3 — 7 → 1 | ✅ PASSED |
| Complete TinyNPU — 21 → 14 → 7 → 1 | ✅ PASSED |

---

## Current Status

### Machine Learning

- [x] Load numerical dataset from CSV
- [x] Implement 21 → 14 → 7 → 1 architecture
- [x] Implement ReLU activation
- [x] Implement sigmoid output
- [x] Implement forward propagation
- [x] Implement binary cross-entropy loss
- [x] Implement backpropagation
- [x] Update weights and biases using gradient descent
- [x] Train and test the Python model
- [x] Save trained weights and biases as NumPy `.npy` files

### Quantization

- [x] Implement per-layer INT8 weight quantization
- [x] Implement INT8 input quantization
- [x] Calibrate hidden-layer activation ranges
- [x] Implement INT32 bias quantization
- [x] Generate quantized activation data
- [x] Run quantized Python inference
- [x] Compare FP32 and quantized inference
- [x] Generate reference outputs
- [x] Export weights as `.mem` files
- [x] Export biases as `.mem` files
- [x] Export inputs as `.mem` files
- [x] Export test/reference vectors as `.mem` files

### RTL

- [x] Install and configure Icarus Verilog
- [x] Build MAC unit in SystemVerilog
- [x] Verify MAC unit
- [x] Build ReLU unit
- [x] Verify ReLU unit
- [x] Build requantization unit
- [x] Verify requantization unit
- [x] Build single-neuron datapath
- [x] Verify single neuron
- [x] Build Layer 1 — 21 → 14
- [x] Verify Layer 1
- [x] Build Layer 2 — 14 → 7
- [x] Verify Layer 2
- [x] Build Layer 3 — 7 → 1
- [x] Verify Layer 3
- [x] Build complete TinyNPU top-level
- [x] Verify complete 21 → 14 → 7 → 1 RTL integration

### Real Model RTL Verification

- [ ] Load actual `.mem` files into RTL
- [ ] Feed actual quantized inputs into RTL
- [ ] Use actual trained weights
- [ ] Use actual quantized biases
- [ ] Execute the trained model inside RTL
- [ ] Capture the RTL prediction
- [ ] Compare RTL output with Python quantized inference
- [ ] Verify predicted classes match
- [ ] Automate Python-vs-RTL verification

### FPGA

- [ ] Add hardware memory/ROM interfaces
- [ ] Add `$readmemh()` model loading
- [ ] Add start/done control
- [ ] Add control FSM
- [ ] Add layer sequencing
- [ ] Synthesize for an FPGA
- [ ] Measure LUT usage
- [ ] Measure FF usage
- [ ] Measure DSP usage
- [ ] Measure BRAM usage
- [ ] Measure maximum clock frequency
- [ ] Measure latency
- [ ] Measure throughput
- [ ] Measure power

### Optimization

- [ ] Explore sequential vs. parallel MAC architectures
- [ ] Explore resource sharing
- [ ] Explore parallel neuron execution
- [ ] Add pipelining
- [ ] Optimize memory organization
- [ ] Optimize clock frequency
- [ ] Optimize latency
- [ ] Optimize throughput
- [ ] Analyze area/performance trade-offs

---

## Roadmap

### Phase 1 — Machine Learning

Train and validate the Python neural network.

**Status: COMPLETE**

---

### Phase 2 — Quantization

Convert the floating-point neural network into an INT8/INT32 representation suitable for hardware inference.

**Status: COMPLETE**

---

### Phase 3 — Quantized Inference

Run inference using the quantized model and compare the result against the original FP32 model.

**Status: COMPLETE**

---

### Phase 4 — Hardware Data Export

Export the quantized inputs, weights, biases, test vectors, and scale information into hardware-friendly files.

**Status: COMPLETE**

---

### Phase 5 — RTL Datapath

Implement the fundamental neural-network hardware operations:

```text
MAC
 ↓
Requantization
 ↓
ReLU
 ↓
Neuron
 ↓
Layer
 ↓
Complete Network
```

**Status: COMPLETE**

---

### Phase 6 — RTL Verification

Verify each component, each neural-network layer, and the complete network.

```text
MAC
 ↓
ReLU
 ↓
Requantization
 ↓
Neuron
 ↓
Layer 1
 ↓
Layer 2
 ↓
Layer 3
 ↓
TinyNPU
```

**Status: COMPLETE**

---

### Phase 7 — Real Model RTL Verification

Connect the actual quantized model generated by Python to the SystemVerilog implementation.

```text
Python Training
       ↓
Quantization
       ↓
.mem Files
       ↓
RTL Memory
       ↓
TinyNPU
       ↓
RTL Prediction
       ↓
Python Reference
       ↓
PASS / FAIL
```

**Status: NEXT**

---

### Phase 8 — FPGA Implementation

Deploy the verified RTL design onto an FPGA.

The FPGA implementation will be evaluated using:

- LUTs
- Flip-Flops
- DSP blocks
- BRAM
- Maximum clock frequency
- Latency
- Throughput
- Power

---

### Phase 9 — Hardware Optimization

Explore different accelerator architectures and evaluate their trade-offs.

Potential experiments include:

- Sequential MAC architecture
- Parallel MAC architecture
- Resource sharing
- Parallel neuron execution
- Pipelining
- Memory organization
- Clock frequency optimization
- Latency optimization
- Throughput optimization
- Area/performance trade-offs

---

## Why TinyNPU?

TinyNPU is designed to demonstrate the connection between several areas that are often studied independently:

```text
Machine Learning
       +
Neural Network Mathematics
       +
Backpropagation
       +
Quantization
       +
Fixed-Point Arithmetic
       +
Digital Logic
       +
SystemVerilog
       +
RTL Verification
       +
FPGA Architecture
       +
Hardware Optimization
```

The project demonstrates how a model trained in software can eventually become a hardware inference accelerator.

Instead of treating machine learning and digital hardware as separate topics, TinyNPU connects them through a single end-to-end implementation.

---

## Training vs Inference

A key design decision in TinyNPU is the separation between training and inference.

Training occurs in Python:

```text
Dataset
   ↓
Forward Pass
   ↓
Loss
   ↓
Backpropagation
   ↓
Gradient Descent
   ↓
Updated Parameters
```

Once training is complete, the parameters are frozen.

The hardware performs only inference:

```text
Quantized Input
      ↓
TinyNPU
      ↓
MAC
      ↓
Requantization
      ↓
ReLU
      ↓
Next Layer
      ↓
Prediction
```

This reflects the typical hardware-acceleration approach where expensive model training remains in software while inference is accelerated using dedicated hardware.

---

## Icarus Verilog Simulation

TinyNPU currently uses Icarus Verilog for RTL compilation and simulation.

Example compilation command:

```cmd
C:\iverilog\bin\iverilog.exe -g2012 -o tinynpu_sim rtl/neuron.sv rtl/layer1.sv rtl/layer2.sv rtl/layer3.sv rtl/tinynpu.sv rtl/tb_tinynpu.sv
```

Run the compiled simulation:

```cmd
C:\iverilog\bin\vvp.exe tinynpu_sim
```

The successful integration simulation reports:

```text
====================================
TinyNPU Complete Network Test
====================================

Architecture:
21 -> 14 -> 7 -> 1

FINAL OUTPUT CHECK PASSED

====================================
TINY NPU INTEGRATION TEST PASSED
21 -> 14 -> 7 -> 1
====================================
```

---

## Important Verification Note

The current RTL integration test verifies that the hardware architecture and datapath are functioning correctly with deterministic test values.

It should not yet be interpreted as proof that the trained Python model has been reproduced exactly in RTL.

The next major milestone is to replace the deterministic RTL testbench values with the actual exported model data:

```text
inputs.mem
weights1.mem
bias1.mem
weights2.mem
bias2.mem
weights3.mem
bias3.mem
```

The final verification flow will be:

```text
              Python
                │
                ▼
        Trained FP32 Model
                │
                ▼
          Quantization
                │
                ▼
             .mem
                │
                ▼
        SystemVerilog RTL
                │
                ▼
          TinyNPU Output
                │
                ▼
             Compare
                │
        ┌───────┴───────┐
        ▼               ▼
      PASS              FAIL
```

---

## Project Goal

The ultimate goal of TinyNPU is to demonstrate the complete journey of a neural-network model:

```text
                 MACHINE LEARNING
                        │
                        ▼
                 Python Training
                        │
                        ▼
                   FP32 Model
                        │
                        ▼
                  Quantization
                        │
                        ▼
                   INT8 / INT32
                        │
                        ▼
                    .mem Files
                        │
                        ▼
                SystemVerilog RTL
                        │
                        ▼
                  RTL Simulation
                        │
                        ▼
              Python vs RTL Verification
                        │
                        ▼
                    FPGA NPU
                        │
                        ▼
              Hardware Optimization
```

The project aims to demonstrate how neural-network computation can be translated from a software model into a hardware inference accelerator.

---

## Disclaimer

The current dataset is a small toy dataset intended for validating the machine-learning and hardware pipeline.

It is **not** sufficient to demonstrate meaningful real-world model generalization or production-level accuracy.

The reported accuracy values are therefore only useful for validating the current implementation and test vectors.

The primary objective of TinyNPU is to explore the engineering path from:

**Machine Learning → Quantization → Fixed-Point Arithmetic → RTL → FPGA**

rather than to develop a production-grade machine-learning model.

---

## Author

**Pranav Kamble**

Built as an exploration of:

**ML + Quantization + SystemVerilog + RTL Verification + FPGA Hardware Acceleration**

```text
Python
   +
Neural Networks
   +
Quantization
   +
Fixed-Point Arithmetic
   +
SystemVerilog
   +
RTL Verification
   +
FPGA
   =
TinyNPU
```
