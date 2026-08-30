# TinyNPU

**TinyNPU** is a learning-focused neural-network inference accelerator project that explores the path from **machine learning in Python to hardware implementation using RTL and FPGA technology**.

The project currently starts with a small feed-forward neural network:

```text
21 → 14 → 7 → 1
```

The long-term goal is to train the network in Python, quantize the learned parameters, export them into hardware-friendly memory files, implement the inference engine in SystemVerilog, verify the RTL against the Python reference model, and evaluate FPGA performance/resource usage.

---

## Project Architecture

```text
                  DATASET
                     │
                     ▼
              ┌─────────────┐
              │    Python   │
              │  Data Load  │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │   Training  │
              │ Backpropagation
              └──────┬──────┘
                     │
              Learned Parameters
                     │
                     ▼
              ┌─────────────┐
              │ Quantization│
              │   / Export  │
              └──────┬──────┘
                     │
                  .mem files
                     │
                     ▼
              ┌─────────────┐
              │ SystemVerilog
              │ RTL Inference
              └──────┬──────┘
                     │
                     ▼
                   FPGA
```

---

## Neural Network

The current model is a fully connected feed-forward neural network:

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

### Trainable parameters

| Layer | Shape | Weights | Biases |
|---|---:|---:|---:|
| 21 → 14 | `(14, 21)` | 294 | 14 |
| 14 → 7 | `(7, 14)` | 98 | 7 |
| 7 → 1 | `(1, 7)` | 7 | 1 |
| **Total** | | **399** | **22** |

Total trainable parameters: **421**.

---

## Current Status

### Completed

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

### In Progress / Planned

- [ ] Use a larger, meaningful dataset
- [ ] Properly quantize the trained model to INT8
- [ ] Export weights, biases, inputs, and expected outputs as `.mem` files
- [ ] Build MAC unit in SystemVerilog
- [ ] Build neuron/layer RTL
- [ ] Build complete 21 → 14 → 7 → 1 inference engine
- [ ] Create an RTL testbench
- [ ] Automatically compare Python and RTL outputs
- [ ] Add pipelining and parallelism
- [ ] Synthesize for an FPGA
- [ ] Measure LUTs, FFs, DSPs, BRAM, timing, latency, throughput, and power
- [ ] Explore area/performance trade-offs

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
│
├── tinynpu_21_feature_dataset.csv
│
├── requirements.txt
│
├── rtl/                       # Planned RTL implementation
│   ├── mac.sv
│   ├── neuron.sv
│   ├── layer.sv
│   └── tb.sv
│
└── README.md
```

Generated files such as trained `.npy` parameters and `.mem` hardware-memory files are intentionally excluded from Git.

---

## Python Environment

The project uses Python with NumPy and Pandas.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the current Python pipeline:

```bash
python main.py
```

---

## Training Flow

The Python side follows:

```text
CSV
 ↓
Load features + targets
 ↓
Initialize weights
 ↓
Forward propagation
 ↓
Calculate loss
 ↓
Backpropagation
 ↓
Update weights/biases
 ↓
Repeat for multiple epochs
 ↓
Save trained parameters
```

The trained parameters are the bridge between the ML side and the future hardware implementation.

---

## Hardware Flow

The planned FPGA flow is:

```text
Trained FP32 model
       ↓
INT8 quantization
       ↓
Export weights / biases
       ↓
.mem files
       ↓
SystemVerilog $readmemh()
       ↓
FPGA memory
       ↓
MAC operations
       ↓
ReLU / Sigmoid
       ↓
Prediction
```

The FPGA will perform **inference**, not training.

Training and backpropagation remain in Python, while the FPGA will use the learned, frozen parameters to perform inference efficiently.

---

## Verification Strategy

Python will act as the **golden reference model**.

The same inputs and quantized parameters will be supplied to:

```text
Python Reference
       │
       ├──────────────┐
       │              │
       ▼              ▼
Expected Output    RTL Output
       │              │
       └──────┬───────┘
              ▼
          Comparator
              │
          PASS / FAIL
```

The goal is to verify that the RTL implementation produces results consistent with the Python model under the same fixed-point arithmetic rules.

---

## Why TinyNPU?

This project is intended to demonstrate the complete connection between:

- Machine learning
- Neural-network mathematics
- Backpropagation
- Quantization
- Digital hardware
- RTL/SystemVerilog
- FPGA architecture
- Hardware verification
- Pipelining
- Parallelism
- Performance/area optimization

Rather than treating ML and hardware as separate topics, TinyNPU connects them into one end-to-end workflow.

---

## Roadmap

### Phase 1 — ML
Train and validate the Python model.

### Phase 2 — Quantization
Convert floating-point parameters and inputs into an FPGA-friendly fixed-point representation.

### Phase 3 — RTL
Implement the neural-network inference datapath in SystemVerilog.

### Phase 4 — Verification
Compare RTL outputs against the Python golden model.

### Phase 5 — FPGA
Synthesize and deploy the accelerator on an FPGA.

### Phase 6 — Optimization
Experiment with:

- Sequential vs. parallel MACs
- Resource sharing
- Pipelining
- Memory organization
- Clock frequency
- Latency
- Throughput
- LUT/DSP/BRAM utilization

---

## Disclaimer

The current dataset is a small toy dataset intended for validating the learning pipeline. It is **not** sufficient to demonstrate meaningful real-world model generalization or production-level accuracy.

The hardware implementation and FPGA benchmarking are the next stages of the project.

---

## Author

**Pranav Kamble**

Built as an exploration of **ML + RTL + FPGA hardware acceleration**.
