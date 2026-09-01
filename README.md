# TinyNPU

**TinyNPU** is a learning-focused neural-network inference accelerator project that explores the complete path from **machine learning in Python to hardware implementation using SystemVerilog RTL and FPGA technology**.

The project connects a software-based neural-network training pipeline with a hardware-oriented inference implementation.

The current neural network architecture is:

```text
21 → 14 → 7 → 1
```

The project follows the complete workflow:

```text
Dataset
   ↓
Python Neural Network
   ↓
FP32 Training
   ↓
FP32 Inference
   ↓
INT8 / INT32 Quantization
   ↓
Quantized Inference
   ↓
.mem Hardware Data Export
   ↓
SystemVerilog RTL
   ↓
RTL Simulation
   ↓
Python vs RTL Verification
   ↓
FPGA Implementation
```

The Python side is responsible for training and quantization.

The SystemVerilog side is responsible for implementing neural-network inference in hardware.

The ultimate goal is to demonstrate how a trained machine-learning model can be translated into a hardware inference accelerator and eventually evaluated on an FPGA.

---

## Project Architecture

```text
                         DATASET
                            │
                            ▼
                    ┌──────────────┐
                    │    Python    │
                    │  Data Load   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Training   │
                    │ Backpropagation
                    └──────┬───────┘
                           │
                           ▼
                    Learned Parameters
                           │
                           ▼
                    ┌──────────────┐
                    │   FP32 Model │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Quantization │
                    │  INT8/INT32  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Quantized    │
                    │  Inference   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  .mem Export │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ SystemVerilog│
                    │     RTL      │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ RTL Simulation│
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Python vs RTL│
                    │ Verification │
                    └──────┬───────┘
                           │
                           ▼
                          FPGA
```

---

## Neural Network

The current model is a fully connected feed-forward neural network.

```text
Input Layer
21 features
     │
     ▼
Hidden Layer 1
14 neurons
     │
   ReLU
     │
     ▼
Hidden Layer 2
7 neurons
     │
   ReLU
     │
     ▼
Output Layer
1 neuron
     │
  Sigmoid
     │
     ▼
Binary prediction
```

The architecture is:

```text
21 → 14 → 7 → 1
```

### Trainable Parameters

| Layer | Shape | Weights | Biases |
|---|---:|---:|---:|
| 21 → 14 | `(14, 21)` | 294 | 14 |
| 14 → 7 | `(7, 14)` | 98 | 7 |
| 7 → 1 | `(1, 7)` | 7 | 1 |
| **Total** | | **399** | **22** |

Total trainable parameters:

```text
399 weights + 22 biases = 421 parameters
```

---

## Machine Learning Pipeline

The Python implementation performs the complete software-side neural-network workflow.

```text
CSV Dataset
     ↓
Load Features + Targets
     ↓
Initialize Parameters
     ↓
Forward Propagation
     ↓
Calculate Binary Cross-Entropy Loss
     ↓
Backpropagation
     ↓
Calculate Gradients
     ↓
Gradient Descent
     ↓
Update Weights + Biases
     ↓
Repeat for Multiple Epochs
     ↓
Trained FP32 Model
     ↓
FP32 Inference
     ↓
Quantization
     ↓
Quantized Inference
     ↓
Hardware Parameter Export
     ↓
.mem Files
```

The trained parameters form the bridge between the machine-learning implementation and the hardware implementation.

---

## FP32 Model

The initial neural network operates using floating-point values.

The Python model contains:

```text
FP32 Inputs
FP32 Weights
FP32 Biases
FP32 Activations
FP32 Output
```

The model performs:

```text
Layer 1
21 → 14
     ↓
ReLU
     ↓
Layer 2
14 → 7
     ↓
ReLU
     ↓
Layer 3
7 → 1
     ↓
Sigmoid
     ↓
Probability
```

The final sigmoid output represents the binary-classification probability.

---

## Quantization

After training, the floating-point model is converted into an integer representation suitable for hardware inference.

The project uses:

```text
INT8
```

for inputs, weights, and intermediate activations.

Biases and accumulators use:

```text
INT32
```

### Weight Quantization

Weights are converted from:

```text
FP32
 ↓
INT8
```

using scale factors.

Conceptually:

```text
INT8 ≈ FP32 / scale
```

and during reconstruction:

```text
FP32 ≈ INT8 × scale
```

The signed INT8 range is:

```text
-128 → 127
```

Separate scale factors are maintained for the different layers.

### Input Quantization

The model inputs are converted from FP32 to INT8:

```text
FP32 Input
     ↓
Quantization
     ↓
INT8 Input
```

An input scale is exported along with the quantized input data.

### Activation Quantization

The hidden-layer activations are calibrated and converted to INT8.

The hardware-oriented computation therefore becomes:

```text
INT8 Input
    ×
INT8 Weight
    ↓
INT32 Accumulator
    +
INT32 Bias
    ↓
Requantization
    ↓
INT8 Activation
    ↓
ReLU
    ↓
Next Layer
```

### Bias Quantization

Biases are represented using INT32 values.

This provides sufficient numerical range for the accumulated products generated by the MAC operations.

The resulting hardware representation contains:

```text
INT8 Inputs
INT8 Weights
INT32 Biases
INT8 Activations
```

---

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

## RTL Bus Architecture

The implementation uses packed buses to connect the neural-network layers.

### Layer 1 Input

```text
21 × INT8
```

Total bus width:

```text
21 × 8 = 168 bits
```

### Layer 1 Output

```text
14 × INT8
```

Total bus width:

```text
14 × 8 = 112 bits
```

### Layer 2 Output

```text
7 × INT8
```

Total bus width:

```text
7 × 8 = 56 bits
```

### Layer 3 Output

```text
1 × INT8
```

Total bus width:

```text
1 × 8 = 8 bits
```

---

## RTL Modules

The current RTL structure is:

```text
rtl/
│
├── mac.sv
├── relu.sv
├── requantize.sv
├── neuron.sv
│
├── layer1.sv
├── layer2.sv
├── layer3.sv
│
├── tinynpu.sv
│
├── tb_mac.sv
├── tb_relu.sv
├── tb_requantize.sv
├── tb_neuron.sv
├── tb_layer1.sv
├── tb_layer2.sv
├── tb_layer3.sv
└── tb_tinynpu.sv
```

---

## RTL Components

### MAC Unit

The MAC unit performs:

```text
Input × Weight
      +
    Bias
      ↓
INT32 Accumulator
```

Example verification:

```text
Input 0 = 10
Weight 0 = 5

Input 1 = 3
Weight 1 = 2

Input 2 = -4
Weight 2 = 7

Bias = 10

Expected result = 38
RTL result      = 38
```

Result:

```text
TEST PASSED
```

---

### ReLU Unit

The ReLU hardware implements:

```text
ReLU(x) = max(0, x)
```

Test cases included:

```text
-10 → 0
-1  → 0
 0  → 0
 5  → 5
127 → 127
```

Result:

```text
ReLU Test Complete
```

Status:

```text
PASSED
```

---

### Requantization Unit

The requantization unit converts the INT32 accumulator back to an INT8 representation.

It was tested using:

```text
0
positive accumulator
negative accumulator
large positive accumulator
large negative accumulator
```

The implementation also verifies INT8 saturation:

```text
Maximum = 127
Minimum = -128
```

Result:

```text
Requantization Test Complete
```

Status:

```text
PASSED
```

---

### Single Neuron

The single-neuron RTL integrates:

```text
MAC
 ↓
Requantization
 ↓
ReLU
```

The complete neuron datapath was verified independently.

Result:

```text
NEURON TEST PASSED
```

Status:

```text
PASSED
```

---

## Layer 1

Layer 1 implements:

```text
21 → 14
```

It contains 14 neurons.

The layer was tested independently.

Verification result:

```text
LAYER 1 TEST PASSED
14/14 NEURONS PASSED
```

Status:

```text
PASSED
```

---

## Layer 2

Layer 2 implements:

```text
14 → 7
```

It contains seven neurons.

Verification result:

```text
LAYER 2 TEST PASSED
7/7 NEURONS PASSED
```

Status:

```text
PASSED
```

---

## Layer 3

Layer 3 implements:

```text
7 → 1
```

It contains one output neuron.

Verification result:

```text
LAYER 3 TEST PASSED
```

Status:

```text
PASSED
```

---

## Complete TinyNPU RTL Integration

The individual layers have been integrated into the complete neural-network architecture:

```text
21 → 14 → 7 → 1
```

The hierarchy is:

```text
Layer 1
   ↓
Layer 2
   ↓
Layer 3
   ↓
Final Output
```

The complete RTL integration simulation successfully passed:

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

Status:

```text
PASSED
```

The current integration test uses deterministic test values to verify the hardware hierarchy and datapath.

It is not yet the final verification of the trained neural-network parameters.

---

## Hardware Memory Export

The quantized Python pipeline exports the hardware parameters into `.mem` files.

Current memory structure:

```text
mem/
│
├── inputs.mem
│
├── weights1.mem
├── bias1.mem
│
├── weights2.mem
├── bias2.mem
│
├── weights3.mem
├── bias3.mem
│
├── targets.mem
├── expected_classes.mem
│
└── scales.txt
```

### Memory Contents

#### Inputs

```text
inputs.mem
```

Contains:

```text
INT8 input values
```

Current test data:

```text
2 samples × 21 features
```

Total:

```text
42 INT8 values
```

#### Layer 1

```text
weights1.mem
bias1.mem
```

Contains:

```text
294 INT8 weights
14 INT32 biases
```

#### Layer 2

```text
weights2.mem
bias2.mem
```

Contains:

```text
98 INT8 weights
7 INT32 biases
```

#### Layer 3

```text
weights3.mem
bias3.mem
```

Contains:

```text
7 INT8 weights
1 INT32 bias
```

#### Test Data

```text
targets.mem
expected_classes.mem
```

These files provide reference information for validating predictions.

#### Scale Information

```text
scales.txt
```

Contains the quantization scale information required to interpret the fixed-point model parameters.

---

## `.mem` Export Flow

The Python hardware export stage follows:

```text
Quantized NumPy Arrays
        ↓
Convert Integer Values
        ↓
Format as Hardware Memory Values
        ↓
Write .mem Files
        ↓
SystemVerilog $readmemh()
```

The `.mem` files are intended to become the interface between the Python quantization pipeline and the RTL hardware implementation.

---

## Python Files

The Python side currently contains scripts for:

```text
main.py
quantize.py
quantized_inference.py
export_mem.py
```

along with the other project modules used by the training pipeline.

The main responsibilities are:

```text
main.py
    ↓
Training + Pipeline Coordination

quantize.py
    ↓
INT8 / INT32 Quantization

quantized_inference.py
    ↓
Quantized Model Inference

export_mem.py
    ↓
Hardware .mem Generation
```

---

## Repository Structure

```text
TinyNPU/
│
├── main.py
├── model.py
├── backprop.py
├── load.py
├── export_weights.py
├── quantize.py
├── quantized_inference.py
├── export_mem.py
├── create_docs.py
│
├── tinynpu_21_feature_dataset.csv
│
├── requirements.txt
│
├── rtl/
│   ├── mac.sv
│   ├── relu.sv
│   ├── requantize.sv
│   ├── neuron.sv
│   │
│   ├── layer1.sv
│   ├── layer2.sv
│   ├── layer3.sv
│   ├── tinynpu.sv
│   │
│   ├── tb_mac.sv
│   ├── tb_relu.sv
│   ├── tb_requantize.sv
│   ├── tb_neuron.sv
│   ├── tb_layer1.sv
│   ├── tb_layer2.sv
│   ├── tb_layer3.sv
│   └── tb_tinynpu.sv
│
├── quantized/
│   ├── inputs_int8.npy
│   ├── input_scale.txt
│   ├── weights1_int8.npy
│   ├── weights1_scale.txt
│   ├── weights2_int8.npy
│   ├── weights2_scale.txt
│   ├── weights3_int8.npy
│   ├── weights3_scale.txt
│   ├── activation1_int8.npy
│   ├── activation1_scale.txt
│   ├── activation2_int8.npy
│   ├── activation2_scale.txt
│   ├── bias1_int32.npy
│   ├── bias1_scale.txt
│   ├── bias2_int32.npy
│   ├── bias2_scale.txt
│   ├── bias3_int32.npy
│   ├── bias3_scale.txt
│   ├── outputs_fp32.npy
│   └── targets_int8.npy
│
├── mem/
│   ├── inputs.mem
│   ├── weights1.mem
│   ├── bias1.mem
│   ├── weights2.mem
│   ├── bias2.mem
│   ├── weights3.mem
│   ├── bias3.mem
│   ├── targets.mem
│   ├── expected_classes.mem
│   └── scales.txt
│
└── README.md
```

Generated model artifacts such as `.npy` files and hardware `.mem` files may be excluded from Git depending on repository size and project requirements.

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
