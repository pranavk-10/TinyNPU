# TinyNPU: End-to-End Deep Learning Hardware Accelerator
## Complete Engineering & Mathematical Reference Manual

---

### Executive Overview & Project Mission

**TinyNPU** is an end-to-end, educational neural network inference accelerator project designed to bridge the gap between high-level Machine Learning (ML) software frameworks and custom Digital Logic design in **SystemVerilog RTL** targeting **Field Programmable Gate Arrays (FPGAs)** and **ASICs**.

Modern deep learning is typically developed in high-level Python environments using floating-point representations (FP32 / FP16). However, general-purpose CPUs and GPUs consume substantial power and area when executing floating-point tensor operations. Custom Neural Processing Units (NPUs) solve this bottleneck by:
1. Translating neural models into **fixed-point integer arithmetic (INT8 / INT32)**.
2. Hardwiring matrix-vector Multiply-Accumulate (MAC) datapaths in dedicated digital logic.
3. Eliminating software instruction overhead and streaming tensors directly through spatial systolic or combinational arrays.

This project demonstrates every step of this journey:
$$\text{CSV Dataset} \longrightarrow \text{FP32 Training} \longrightarrow \text{Quantization} \longrightarrow \text{.mem Export} \longrightarrow \text{SystemVerilog RTL} \longrightarrow \text{Bit-Accurate Verification} \longrightarrow \text{FPGA Deployment}$$

---

## 1. Network Topology & Machine Learning Mathematics

### 1.1 Architecture Specification

The network is a Multilayer Perceptron (MLP) binary classifier configured as **$21 \longrightarrow 14 \longrightarrow 7 \longrightarrow 1$**:

```
Input Vector x ∈ ℝ²¹
       │
       ▼  [W¹: 14 × 21,  b¹: 14]  (294 weights, 14 biases)
Layer 1: 14 Neurons + ReLU Activation
       │
       ▼  [W²: 7 × 14,   b²: 7]   (98 weights, 7 biases)
Layer 2: 7 Neurons + ReLU Activation
       │
       ▼  [W³: 1 × 7,    b³: 1]   (7 weights, 1 bias)
Layer 3: 1 Output Neuron + Sigmoid Activation
       │
       ▼
Binary Classification Output ŷ ∈ (0, 1)
```

**Total Trainable Parameters:**
$$\text{Weights} = (21 \times 14) + (14 \times 7) + (7 \times 1) = 294 + 98 + 7 = 399$$
$$\text{Biases} = 14 + 7 + 1 = 22$$
$$\text{Total Parameters} = 399 + 22 = 421$$

---

### 1.2 Forward Propagation Mathematics

For an input feature vector $\mathbf{x} \in \mathbb{R}^{21 \times 1}$:

#### Layer 1 (Hidden Layer 1):
$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)} \quad \in \mathbb{R}^{14 \times 1}$$
$$\mathbf{h}^{(1)} = \text{ReLU}\left(\mathbf{z}^{(1)}\right) = \max\left(0, \mathbf{z}^{(1)}\right)$$

#### Layer 2 (Hidden Layer 2):
$$\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{h}^{(1)} + \mathbf{b}^{(2)} \quad \in \mathbb{R}^{7 \times 1}$$
$$\mathbf{h}^{(2)} = \text{ReLU}\left(\mathbf{z}^{(2)}\right) = \max\left(0, \mathbf{z}^{(2)}\right)$$

#### Layer 3 (Output Layer):
$$z^{(3)} = \mathbf{W}^{(3)} \mathbf{h}^{(2)} + b^{(3)} \quad \in \mathbb{R}^{1 \times 1}$$
$$\hat{y} = \sigma\left(z^{(3)}\right) = \frac{1}{1 + e^{-z^{(3)}}}$$

#### Classification Rule:
$$\text{Class}(\mathbf{x}) = \begin{cases} 1 & \text{if } \hat{y} \ge 0.5 \iff z^{(3)} \ge 0 \\ 0 & \text{if } \hat{y} < 0.5 \iff z^{(3)} < 0 \end{cases}$$

---

### 1.3 Loss Function: Binary Cross-Entropy (BCE)

For a ground truth binary label $y \in \{0, 1\}$ and model prediction $\hat{y} \in (0, 1)$:
$$\mathcal{L}(y, \hat{y}) = -\left[ y \ln(\hat{y} + \epsilon) + (1 - y) \ln(1 - \hat{y} + \epsilon) \right]$$
*(where $\epsilon = 10^{-8}$ is a numerical stability term preventing $\ln(0)$).*

---

### 1.4 Backpropagation Calculus & Gradient Descent

Training calculates the partial derivatives of the loss with respect to all weights and biases via the chain rule.

#### 1. Output Layer Gradients ($\mathbf{W}^{(3)}, b^{(3)}$):
Since $\hat{y} = \sigma(z^{(3)})$ and $\sigma'(z) = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$:
$$\delta^{(3)} = \frac{\partial \mathcal{L}}{\partial z^{(3)}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{(3)}} = \left( -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} \right) \cdot \hat{y}(1-\hat{y}) = \hat{y} - y$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(3)}} = \delta^{(3)} \cdot (\mathbf{h}^{(2)})^T \quad \in \mathbb{R}^{1 \times 7}, \qquad \frac{\partial \mathcal{L}}{\partial b^{(3)}} = \delta^{(3)} \quad \in \mathbb{R}$$

#### 2. Layer 2 Gradients ($\mathbf{W}^{(2)}, \mathbf{b}^{(2)}$):
$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(2)}} = (\mathbf{W}^{(3)})^T \delta^{(3)} \quad \in \mathbb{R}^{7 \times 1}$$
Since $\text{ReLU}'(z) = \mathbb{I}(z > 0)$:
$$\boldsymbol{\delta}^{(2)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(2)}} \odot \mathbb{I}\left(\mathbf{z}^{(2)} > 0\right) = \left[(\mathbf{W}^{(3)})^T \delta^{(3)}\right] \odot \mathbb{I}\left(\mathbf{z}^{(2)} > 0\right)$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(2)}} = \boldsymbol{\delta}^{(2)} \cdot (\mathbf{h}^{(1)})^T \quad \in \mathbb{R}^{7 \times 14}, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(2)}} = \boldsymbol{\delta}^{(2)} \quad \in \mathbb{R}^{7 \times 1}$$

#### 3. Layer 1 Gradients ($\mathbf{W}^{(1)}, \mathbf{b}^{(1)}$):
$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(1)}} = (\mathbf{W}^{(2)})^T \boldsymbol{\delta}^{(2)} \quad \in \mathbb{R}^{14 \times 1}$$
$$\boldsymbol{\delta}^{(1)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}} = \left[(\mathbf{W}^{(2)})^T \boldsymbol{\delta}^{(2)}\right] \odot \mathbb{I}\left(\mathbf{z}^{(1)} > 0\right)$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \boldsymbol{\delta}^{(1)} \cdot \mathbf{x}^T \quad \in \mathbb{R}^{14 \times 21}, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(1)}} = \boldsymbol{\delta}^{(1)} \quad \in \mathbb{R}^{14 \times 1}$$

#### 4. Parameter Update Rule:
For learning rate $\eta = 0.01$:
$$\mathbf{W}^{(l)} \longleftarrow \mathbf{W}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}, \qquad \mathbf{b}^{(l)} \longleftarrow \mathbf{b}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} \quad \text{for } l \in \{1, 2, 3\}$$

---

## 2. Quantization & Fixed-Point Theory

### 2.1 Why Quantization is Mandatory for Hardware

| Metric | IEEE-754 FP32 Floating Point | Signed INT8 / INT32 Integer |
|---|---|---|
| **Multiplier Hardware Cost** | ~500+ FPGA LUTs per 32-bit float multiplier | 1 DSP48 block can perform two 8-bit multiplies |
| **Power Consumption** | High ($O(N^2)$ exponent/mantissa normalization) | Minimal integer adder/multiplier logic |
| **Memory Bandwidth** | 32 bits (4 bytes) per weight | 8 bits (1 byte) per weight ($4\times$ smaller) |
| **Storage Requirement** | 421 params $\times$ 4 B = 1,684 Bytes | 399 $\times$ 1 B + 22 $\times$ 4 B = 487 Bytes |

---

### 2.2 Symmetrical Affine Quantization

We map continuous real values $x \in \mathbb{R}$ to signed 8-bit integers $q \in [-128, 127]$:

$$S = \frac{\max\left(|X|\right)}{127}$$
$$q = \text{clip}\left(\left\lfloor \frac{x}{S} \right\rceil, -128, 127\right)$$
$$x \approx \tilde{x} = q \times S$$

where:
- $S$ is the **Scale Factor** (floating-point scalar representing the value of 1 integer LSB).
- $\lfloor \cdot \rceil$ is round-to-nearest integer.

---

### 2.3 Why Biases and Accumulators Use INT32

When multiplying two signed INT8 numbers:
$$\text{INT8} \times \text{INT8} \longrightarrow \text{INT16}$$
$$(-128 \dots 127) \times (-128 \dots 127) = -16,129 \dots +16,384 \quad \text{(fits in 16 bits)}$$

When summing $N$ products together in a neuron dot product:
$$\text{acc} = \sum_{i=0}^{N-1} (x_i \cdot w_i)$$
The maximum dynamic range expands by $\log_2(N)$ bits:
$$\text{Bits Required} = 16 + \lceil \log_2(N) \rceil$$
For $N = 21$: $16 + \lceil 4.39 \rceil = 21\text{ bits}$.

To prevent integer overflow in silicon while aligning with standard computing words, **32-bit signed accumulators (`logic signed [31:0]`)** are used.

#### Bias Quantization Rule:
The dot product accumulator has scale $S_{\text{acc}} = S_{\text{in}} \cdot S_w$. To add the bias $b$ directly to the accumulator without scale conversion:
$$b_{\text{int32}} = \left\lfloor \frac{b_{\text{fp32}}}{S_{\text{in}} \cdot S_w} \right\rceil$$

---

### 2.4 Fixed-Point Requantization & Dyadic Scaling

After completing an INT32 accumulation, the result must be converted back to an INT8 activation for the next layer.

$$\text{Real Value} = \text{accumulator} \times (S_{\text{in}} \cdot S_w)$$
$$\text{Next Layer INT8} = \frac{\text{Real Value}}{S_{\text{out}}} = \text{accumulator} \times \left( \frac{S_{\text{in}} \cdot S_w}{S_{\text{out}}} \right)$$

Let the scale ratio be $M = \frac{S_{\text{in}} \cdot S_w}{S_{\text{out}}}$. Because $M < 1$, floating-point division is required in software.

In hardware, we avoid floating point division entirely by using **Dyadic Fixed-Point Scaling**:
$$M \approx \frac{\text{MULTIPLIER}}{2^{\text{SHIFT}}}$$
$$\text{MULTIPLIER} = \left\lfloor M \cdot 2^{\text{SHIFT}} \right\rceil, \quad \text{where } \text{SHIFT} = 20$$

In SystemVerilog RTL, this requires only an integer multiply and an arithmetic right shift (`>>>`):
```systemverilog
scaled_value  = accumulator * MULTIPLIER;       // 32-bit * 32-bit -> 64-bit
shifted_value = scaled_value >>> SHIFT;         // Arithmetic right shift by 20

// Clamping / Saturation to INT8
if (shifted_value > 127)
    quantized_value = 8'sd127;
else if (shifted_value < -128)
    quantized_value = -8'sd128;
else
    quantized_value = shifted_value[7:0];
```

---

## 3. Data Format & Hardware Memory Files

### 3.1 Why `.mem` Files and `$readmemh`?

Hardware simulation tools (like Icarus Verilog, ModelSim, Vivado) and FPGA synthesis tools read memory initializations via `$readmemh()` (Hexadecimal memory format) or `$readmemb()` (Binary).

The `.mem` format stores each signed two's-complement value as a newline-delimited hexadecimal string:

| Value Type | Bit Width | Two's Complement Hex Format | Example Values |
|---|---|---|---|
| **INT8 Weights / Inputs** | 8 bits | 2 Hex Digits (`%02X`) | $+1 \to \text{01}$, $+127 \to \text{7F}$, $-1 \to \text{FF}$, $-128 \to \text{80}$ |
| **INT32 Biases** | 32 bits | 8 Hex Digits (`%08X`) | $+17683 \to \text{00004513}$, $-6910 \to \text{FFFFE502}$ |

### 3.2 Directory Layout of Data Artifacts

```
data/
├── inputs/
│   └── tinynpu_21_feature_dataset.csv     # Raw dataset
├── quantized/
│   ├── inputs_int8.npy / input_scale.txt
│   ├── weights1_int8.npy / weights1_scale.txt
│   ├── bias1_int32.npy / bias1_scale.txt
│   ├── ...
├── mem/
│   ├── tinynpu_params.vh                  # Generated SystemVerilog parameters
│   ├── inputs.mem                         # 42 entries (2 samples × 21 features)
│   ├── weights1.mem                       # 294 entries (14 × 21 INT8 weights)
│   ├── bias1.mem                          # 14 entries (14 INT32 biases)
│   ├── weights2.mem                       # 98 entries (7 × 14 INT8 weights)
│   ├── bias2.mem                          # 7 entries (7 INT32 biases)
│   ├── weights3.mem                       # 7 entries (1 × 7 INT8 weights)
│   ├── bias3.mem                          # 1 entry (1 INT32 bias)
│   ├── expected_classes.mem               # Target classification reference
│   ├── layer1_expected.mem                # Golden Layer 1 neuron outputs
│   └── layer2_expected.mem                # Golden Layer 2 neuron outputs
```

---

## 4. SystemVerilog RTL Hardware Implementation

### 4.1 Why SystemVerilog?

1. **`logic signed` Types:** SystemVerilog provides native signed arithmetic support, eliminating manual two's-complement sign extension bugs.
2. **Packed Multi-Dimensional Buses:** Allows passing wide parallel tensor arrays (e.g. 2352-bit weight buses) cleanly across module boundaries without unpacked port limitations in standard toolchains.
3. **`always_comb` Semantics:** Enforces pure combinational intent; EDA tools automatically generate warnings if accidental latches are inferred.
4. **Parameterized Generation (`generate` / `genvar`):** Allows scaling neuron counts and feature dimensions dynamically via parameters.

---

### 4.2 Hardware Datapath Hierarchy

```
                               ┌──────────────────────────────────────────────┐
                               │                 tinynpu.sv                   │
                               │           Top-Level NPU Engine               │
                               └──────────────────────┬───────────────────────┘
                                                      │
                       ┌──────────────────────────────┼──────────────────────────────┐
                       ▼                              ▼                              ▼
        ┌─────────────────────────────┐┌─────────────────────────────┐┌─────────────────────────────┐
        │          layer1.sv          ││          layer2.sv          ││          layer3.sv          │
        │      (14 Neuron Array)      ││      (7 Neuron Array)       ││      (1 Neuron Output)      │
        └──────────────┬──────────────┘└──────────────┬──────────────┘└──────────────┬──────────────┘
                       │                              │                              │
                       ▼                              ▼                              ▼
        ┌─────────────────────────────┐┌─────────────────────────────┐┌─────────────────────────────┐
        │          neuron.sv          ││          neuron.sv          ││          neuron.sv          │
        │   MAC → Requantize → ReLU   ││   MAC → Requantize → ReLU   ││   MAC → Requantize → ReLU   │
        └──────────────┬──────────────┘└──────────────┬──────────────┘└──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
 ┌──────────────┐              ┌──────────────┐
 │    mac.sv    │              │ requantize.sv│
 │  Multiplier- │              │  Fixed-Point │
 │  Accumulator │              │    Scaler    │
 └──────────────┘              └──────────────┘
```

---

### 4.3 Atomic Module Breakdown

#### 1. Multiply-Accumulate (`mac.sv`)
Computes the vector dot product of $N$ INT8 inputs and $N$ INT8 weights, added to an INT32 bias:
$$\text{result} = \text{bias} + \sum_{i=0}^{N-1} (x[i] \cdot w[i])$$

```systemverilog
module mac #(
    parameter integer N = 21
)(
    input  logic signed [7:0]  inputs [0:N-1],
    input  logic signed [7:0]  weights [0:N-1],
    input  logic signed [31:0] bias,
    output logic signed [31:0] result
);
    integer i;
    logic signed [31:0] accumulator;
    logic signed [15:0] product;

    always_comb begin
        accumulator = bias;
        for (i = 0; i < N; i = i + 1) begin
            product = inputs[i] * weights[i];
            accumulator = accumulator + product;
        end
        result = accumulator;
    end
endmodule
```

#### 2. Requantization (`requantize.sv`)
Performs 64-bit intermediate scaling, right-shift, and saturation clamping:
```systemverilog
module requantize #(
    parameter integer MULTIPLIER = 1069,
    parameter integer SHIFT      = 20
)(
    input  logic signed [31:0] accumulator,
    output logic signed [7:0]  output_value
);
    logic signed [63:0] scaled_value;
    logic signed [63:0] shifted_value;

    always_comb begin
        scaled_value  = accumulator * MULTIPLIER;
        shifted_value = scaled_value >>> SHIFT;

        if (shifted_value > 127)
            output_value = 8'sd127;
        else if (shifted_value < -128)
            output_value = -8'sd128;
        else
            output_value = shifted_value[7:0];
    end
endmodule
```

#### 3. ReLU Activation (`relu.sv`)
A zero-latency combinational multiplexer:
```systemverilog
module relu (
    input  logic signed [7:0] input_value,
    output logic signed [7:0] output_value
);
    assign output_value = (input_value < 0) ? 8'sd0 : input_value;
endmodule
```

#### 4. Single Neuron Datapath (`neuron.sv`)
Fuses MAC, Requantize, and ReLU into a unified single-neuron pipeline handling flat packed buses (`[(N*8)-1:0]`).

---

## 5. End-to-End Verification Strategy

### 5.1 Why Testbenches & Automated Verification?

In custom digital design, hardware bugs (e.g. incorrect bit slicing, signed/unsigned mismatches, arithmetic overflow, off-by-one shifts) cannot be debugged with `print()` statements on physical silicon.

**Verification Hierarchy in TinyNPU:**
1. **Unit Testbenches:** Verify individual blocks (`tb_mac.sv`, `tb_relu.sv`, `tb_requantize.sv`, `tb_neuron.sv`).
2. **Layer Testbenches:** Verify parallel neuron arrays (`tb_layer1.sv`, `tb_layer2.sv`, `tb_layer3.sv`).
3. **Integration Testbench:** Verify top-level connectivity (`tb_tinynpu.sv`).
4. **Golden Reference Model Verification (Phase 7):**
   `tb_tinynpu_real.sv` + `tools/verify_rtl.py` loads the exact parameters produced by the Python training run and checks **bit-for-bit equivalence** between Python INT8 inference and SystemVerilog simulation.

```
                    ┌────────────────────────┐
                    │ Python FP32 Reference  │
                    └───────────┬────────────┘
                                │ Quantize
                                ▼
                    ┌────────────────────────┐
                    │ Python INT8 Simulation │
                    └───────────┬────────────┘
                                │ Golden Activations
                                ▼
                    ┌────────────────────────┐
                    │   Comparator Matrix    │ ◄─── SystemVerilog RTL Output
                    │  (tools/verify_rtl.py) │      (tb_tinynpu_real.sv)
                    └───────────┬────────────┘
                                │
                    ┌───────────┴────────────┐
                    ▼                        ▼
               100% MATCH                MISMATCH
             (PARITY PASSED)         (DEBUG ISOLATION)
```

---

## 6. Real Inference Execution Trace (Walkthrough)

### Sample 1 Trace (Ground Truth Target = 0)
- **Input Feature Vector:** $x_{\text{int8}} \in [14, 127]$ (21 features)
- **Layer 1 Output:**
  $$\mathbf{h}^{(1)} = [0, 2, 12, 10, 11, 0, 0, 0, 0, 0, 0, 11, 0, 0]$$
- **Layer 2 Output:**
  $$\mathbf{h}^{(2)} = [0, 0, 0, 15, 0, 0, 0]$$
- **Layer 3 Accumulator:** $\text{acc}^{(3)} = -7561$
- **Layer 3 Requantized & ReLU Output:** $\text{output\_value} = 0$
- **Prediction:** $\text{Class } 0 \quad (\text{Logit } < 0 \implies \hat{y} = 0.1264 < 0.5)$
- **Status:** $\mathbf{MATCH} \checkmark$

### Sample 2 Trace (Ground Truth Target = 1)
- **Input Feature Vector:** $x_{\text{int8}} \in [14, 127]$ (21 features)
- **Layer 1 Output:**
  $$\mathbf{h}^{(1)} = [0, 63, 87, 75, 64, 0, 0, 0, 0, 0, 0, 126, 0, 0]$$
- **Layer 2 Output:**
  $$\mathbf{h}^{(2)} = [63, 126, 37, 17, 0, 80, 0]$$
- **Layer 3 Accumulator:** $\text{acc}^{(3)} = +20117$
- **Layer 3 Requantized & ReLU Output:** $\text{output\_value} = 5$
- **Prediction:** $\text{Class } 1 \quad (\text{Logit } > 0 \implies \hat{y} = 0.9946 \ge 0.5)$
- **Status:** $\mathbf{MATCH} \checkmark$

---

## 7. How to Execute & Verify the Entire Project

### Step 1: Execute Full Python ML & Export Pipeline
```bash
python tools/main.py
```
*Trains model, computes loss, quantizes weights/biases/activations, calculates fixed-point parameters, and exports all `.mem` and `.vh` files into `data/mem/`.*

### Step 2: Run End-to-End Golden Verification
```bash
python tools/verify_rtl.py
```
*Executes the Python bit-accurate hardware emulator and orchestrates Icarus Verilog SystemVerilog simulation to confirm 100% parity across all layers.*

### Step 3: Direct SystemVerilog RTL Simulation
```bash
cd hardware
make real_sim
```
*Compiles all RTL modules and executes `tb_tinynpu_real.sv` in Icarus Verilog (`vvp`).*

---

## 8. Summary of Key Formulas

| Operation | Mathematical Formula | SystemVerilog Equivalent |
|---|---|---|
| **Neuron MAC** | $z = b + \sum_{i=0}^{N-1} x_i w_i$ | `acc = bias + (inputs[i] * weights[i])` |
| **Quantization** | $q = \text{clip}\left(\left\lfloor \frac{x}{S} \right\rceil, -128, 127\right)$ | Precomputed in Python |
| **Fixed-Point Scaler** | $M \approx \frac{\text{MULT}}{2^{\text{SHIFT}}}$ | `scaled = acc * MULT; shifted = scaled >>> SHIFT;` |
| **Saturation Clamping** | $\text{clip}(v, -128, 127)$ | `(v > 127) ? 127 : (v < -128) ? -128 : v[7:0]` |
| **ReLU Activation** | $\max(0, x)$ | `assign out = (in < 0) ? 8'sd0 : in;` |
| **Binary Classification** | $\hat{y} \ge 0.5 \iff z^{(3)} \ge 0$ | `assign class = (output_value > 0) ? 1 : 0;` |

---

*Authored for the TinyNPU Project — Pranav Kamble.*
