# TinyNPU: Complete Project Working

This document explains the complete TinyNPU pipeline, including the neural-network mathematics, numeric formats, quantization formulas, generated files, SystemVerilog datapath, and verification flow.

## 1. Project Purpose

TinyNPU is a small binary-classification neural network implemented twice:

1. As a floating-point and quantized reference model in Python.
2. As an integer neural-network datapath in SystemVerilog RTL.

The purpose is to demonstrate the complete path from a CSV dataset to hardware-ready model parameters and RTL simulation.

The network is:

```text
21 input features -> 14 neurons -> 7 neurons -> 1 output neuron
```

The parameter count is:

```text
Layer 1 weights: 21 x 14 = 294
Layer 1 biases : 14
Layer 2 weights: 14 x 7  = 98
Layer 2 biases : 7
Layer 3 weights: 7 x 1   = 7
Layer 3 bias   : 1
Total          : 421 parameters
```

The dataset currently contains two samples. Therefore, the reported 100% accuracy is only a result for this tiny test set, not a meaningful measure of generalization.

## 2. Repository Structure

```text
TinyNPU/
|
+-- README.md
+-- QUICKSTART.md
+-- PROJECT_STRUCTURE.md
+-- requirements.txt
|
+-- src/
|   +-- load.py
|   +-- model.py
|   +-- backprop.py
|   +-- quantize_model.py
|   +-- quantized_inference.py
|
+-- tools/
|   +-- main.py
|   +-- export_mem.py
|   +-- verify_rtl.py
|   +-- create_architecture_image.py
|
+-- data/
|   +-- inputs/
|   |   +-- tinynpu_21_feature_dataset.csv
|   +-- quantized/
|   |   +-- *.npy
|   |   +-- *_scale.txt
|   +-- mem/
|       +-- *.mem
|       +-- tinynpu_params.vh
|       +-- scales.txt
|
+-- hardware/
    +-- Makefile
    +-- rtl/
        +-- components/
        |   +-- mac.sv
        |   +-- requantize.sv
        |   +-- relu.sv
        +-- layers/
        |   +-- neuron.sv
        |   +-- layer1.sv
        |   +-- layer2.sv
        |   +-- layer3.sv
        +-- top/
        |   +-- tinynpu.sv
        +-- testbenches/
            +-- tb_mac.sv
            +-- tb_requantize.sv
            +-- tb_relu.sv
            +-- tb_neuron.sv
            +-- tb_layer1.sv
            +-- tb_layer2.sv
            +-- tb_layer3.sv
            +-- tb_tinynpu.sv
            +-- tb_tinynpu_real.sv
```

## 3. Complete Data Flow

```text
data/inputs/*.csv
        |
        v
src/load.py
        |
        v
FP32 model: src/model.py
        |
        v
Training: src/backprop.py
        |
        +--> FP32 weights and biases
        |
        v
Quantization: src/quantize_model.py
        |
        +--> data/quantized/*.npy
        +--> data/quantized/*_scale.txt
        |
        v
Memory export: tools/export_mem.py
        |
        +--> data/mem/*.mem
        +--> data/mem/tinynpu_params.vh
        |
        v
RTL simulation: hardware/rtl/
        |
        v
Verification: tools/verify_rtl.py
```

## 4. Input Dataset

The file [data/inputs/tinynpu_21_feature_dataset.csv](../data/inputs/tinynpu_21_feature_dataset.csv) contains 21 feature columns and one `target` column.

`src/load.py` uses pandas to:

1. Read the CSV.
2. Remove the `target` column to create the feature matrix $X$.
3. Keep the `target` column as the label vector $y$.
4. Convert both to NumPy arrays with `float32` values.

The resulting shapes are:

```text
X: (number_of_samples, 21)
y: (number_of_samples,)
```

## 5. Floating-Point Neural Network

The implementation is in [src/model.py](../src/model.py).

### 5.1 Weight Shapes

The weights are initialized using a normal distribution multiplied by `0.1`:

```python
weights1: shape (14, 21)
weights2: shape (7, 14)
weights3: shape (1, 7)
```

Biases start at zero:

```python
bias1: shape (14,)
bias2: shape (7,)
bias3: shape (1,)
```

### 5.2 Layer 1

For input vector $x$ with 21 values:

$$
z_1 = W_1x + b_1$$

where:

- $W_1$ has shape $14 x 21$.
- $x$ has shape $21$.
- $b_1$ has shape $14$.
- $z_1$ has shape $14$.

The hidden activation is ReLU:

$$
h_1 = ReLU(z_1) = max(0, z_1)$$

### 5.3 Layer 2

$$
z_2 = W_2h_1 + b_2$$

where $W_2$ has shape $7 x 14$.

$$
h_2 = ReLU(z_2)$$

### 5.4 Output Layer

$$
z_3 = W_3h_2 + b_3$$

where $W_3$ has shape $1 x 7$.

The Python model converts the output logit into a probability with sigmoid:

$$
\sigma(z_3) = \frac{1}{1 + e^{-z_3}}
$$

The Python classification rule is:

$$
class =
\begin{cases}
1 & \text{if } \sigma(z_3) >= 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

Because $\sigma(z) >= 0.5$ exactly when $z >= 0$, the hardware can classify using the sign of its final integer activation without implementing a floating-point sigmoid.

## 6. Training and Backpropagation

Training is implemented in [src/backprop.py](../src/backprop.py).

The default settings are:

```text
Learning rate: 0.01
Epochs:        1000
Optimizer:     gradient descent
Loss:          binary cross-entropy
```

### 6.1 Binary Cross-Entropy

For target $y$ and predicted probability $p$:

$$
L = -[y \log(p) + (1-y)\log(1-p)]
$$

The code adds a small value $10^{-8}$ inside the logarithms to avoid `log(0)`.

For multiple samples, the printed loss is:

$$
L_{mean} = \frac{1}{N}\sum_{i=1}^{N} L_i
$$

### 6.2 Output-Layer Gradients

For sigmoid followed by binary cross-entropy, the derivative simplifies to:

$$
\frac{\partial L}{\partial z_3} = p-y
$$

The code calls this value `dz3`.

The output-layer gradients are:

$$
\frac{\partial L}{\partial W_3} = dz_3 h_2^T
$$

$$
\frac{\partial L}{\partial b_3} = dz_3
$$

### 6.3 Hidden-Layer Gradients

The gradient entering layer 2 is:

$$
\frac{\partial L}{\partial h_2} = W_3^T dz_3
$$

The ReLU derivative is:

$$
ReLU'(z) =
\begin{cases}
1 & z > 0 \\
0 & z <= 0
\end{cases}
$$

Therefore:

$$
dz_2 = dh_2 \odot ReLU'(z_2)
$$

where $\odot$ means element-by-element multiplication.

Layer 2 gradients:

$$
dW_2 = dz_2 h_1^T$$

$$
db_2 = dz_2$$

The gradient entering layer 1 is:

$$
dh_1 = W_2^T dz_2$$

$$
dz_1 = dh_1 \odot ReLU'(z_1)$$

Layer 1 gradients:

$$
dW_1 = dz_1 x^T$$

$$
db_1 = dz_1$$

### 6.4 Gradient-Descent Update

Every parameter is updated immediately after each training sample:

$$
parameter := parameter - learning\_rate \times gradient
$$

This is online or sample-by-sample gradient descent rather than a single update after the whole dataset.

## 7. Numeric Formats

TinyNPU uses different numeric formats for different jobs.

### 7.1 FP32

FP32 means a 32-bit IEEE floating-point value. It is used for:

- CSV-loaded features after conversion.
- Initial and trained Python weights.
- Python biases.
- Sigmoid probabilities.
- Scale factors.

FP32 is convenient for training because it provides a large dynamic range and avoids early precision loss.

### 7.2 INT8

INT8 is a signed 8-bit integer:

```text
Range: -128 to +127
Width: 8 bits
```

It is used for:

- Inputs.
- Weights.
- Hidden activations.
- RTL neuron outputs.

The real value represented by an INT8 number $q$ is:

$$
real\_value = q \times scale
$$

The quantized value for a floating-point number $r$ is:

$$
q = round\left(\frac{r}{scale}\right)
$$

After rounding, the result is clipped:

$$
q = clip(q, -128, 127)
$$

### 7.3 INT32

INT32 is a signed 32-bit integer:

```text
Range: -2147483648 to +2147483647
Width: 32 bits
```

It is used for:

- MAC accumulators.
- Quantized biases.

INT32 is required because many INT8 products are added together. A single product is at most approximately $127 x 127$, but a layer sums many products.

## 8. Symmetric INT8 Quantization

The function `quantize_int8` in [src/quantize_model.py](../src/quantize_model.py) uses symmetric per-array quantization.

First it finds the largest absolute value:

$$
max\_value = max(|values|)
$$

Then it calculates the scale:

$$
scale = \frac{max\_value}{127}
$$

The value is quantized as:

$$
values\_int8 = round\left(\frac{values}{scale}\right)
$$

Finally it clips the result to the INT8 range and stores it as NumPy `int8`.

If every value is zero, the code uses:

```text
scale = 1.0
quantized value = 0
```

This prevents division by zero.

The same formula is used independently for:

- The input array.
- Layer 1 weights.
- Layer 2 weights.
- Layer 3 weights.
- Layer 1 activations.
- Layer 2 activations.

Each array therefore has its own scale file.

## 9. Quantized Layer Mathematics

Suppose a real input and real weight are represented as:

$$
x_{real} = x_{int8} S_x$$

$$
w_{real} = w_{int8} S_w$$

The real dense-layer calculation is:

$$
z_{real} = \sum_i x_{real,i}w_{real,i} + b_{real}$$

Substituting the quantized values:

$$
z_{real} = \sum_i (x_{int8,i}S_x)(w_{int8,i}S_w) + b_{real}$$

The integer MAC is:

$$
accumulator = \sum_i x_{int8,i}w_{int8,i} + bias_{int32}
$$

The accumulator scale is:

$$
S_{accumulator} = S_x S_w
$$

For hidden-layer output scale $S_y$, the output INT8 value should be:

$$
y_{int8} = round\left(\frac{accumulator \times S_x S_w}{S_y}\right)
$$

This is the mathematical reason that every layer needs the scale of its input, its weights, and its output activation.

## 10. Bias Quantization

A bias is added to an accumulator, so it must use the accumulator's scale.

For a layer receiving values with scale $S_x$ and weights with scale $S_w$:

$$
S_{bias} = S_x S_w
$$

The INT32 bias is:

$$
bias_{int32} = round\left(\frac{bias_{float}}{S_x S_w}\right)
$$

The code clips this result to the INT32 range.

The three layers use different input scales:

```text
Layer 1 bias scale = input_scale      x weights1_scale
Layer 2 bias scale = activation1_scale x weights2_scale
Layer 3 bias scale = activation2_scale x weights3_scale
```

If the quantized bias is converted back to real units:

$$
bias_{real,approx} = bias_{int32} \times S_{bias}
$$

The approximation error comes from rounding to an integer.

## 11. Activation Calibration

The trained FP32 model is run over the available input samples. The hidden activations are collected:

```text
h1 = ReLU(W1 x + b1)
h2 = ReLU(W2 h1 + b2)
```

The arrays of all $h_1$ and $h_2$ values are quantized separately. This produces:

```text
data/quantized/activation1_int8.npy
data/quantized/activation1_scale.txt
data/quantized/activation2_int8.npy
data/quantized/activation2_scale.txt
```

This calibration is based on the samples available during the run. More varied calibration data would be needed for a production model.

## 12. Quantized Python Inference

The reference implementation is [src/quantized_inference.py](../src/quantized_inference.py).

### Layer 1

```text
input_int8 x weights1_int8 -> INT32 dot product
INT32 dot product + bias1_int32
requantize to INT8 using input_scale, weights1_scale, activation1_scale
apply INT8 ReLU
```

Mathematically:

$$
acc_1 = W_{1,int8}x_{int8} + b_{1,int32}
$$

$$
 h_{1,int8} = ReLU\left(round\left(\frac{acc_1 S_{input}S_{W1}}{S_{A1}}\right)\right)
$$

### Layer 2

$$
acc_2 = W_{2,int8}h_{1,int8} + b_{2,int32}
$$

$$
 h_{2,int8} = ReLU\left(round\left(\frac{acc_2 S_{A1}S_{W2}}{S_{A2}}\right)\right)
$$

### Layer 3

The final accumulator is converted back to real units:

$$
z_{3,real} = acc_3 S_{A2}S_{W3}
$$

Then sigmoid produces the reference probability:

$$
p = \frac{1}{1 + e^{-z_{3,real}}}
$$

This gives a floating-point probability that can be compared with the original FP32 model.

## 13. Fixed-Point RTL Requantization

Floating-point division is not used in the RTL. Instead, the scale ratio is approximated by an integer multiplier and power-of-two shift.

The desired operation is:

$$
y = round\left(accumulator \times \frac{S_x S_w}{S_y}\right)
$$

The RTL approximates the scale ratio as:

$$
\frac{S_x S_w}{S_y} \approx \frac{MULTIPLIER}{2^{SHIFT}}
$$

Therefore the hardware calculates:

$$
y = (accumulator \times MULTIPLIER) >>> SHIFT
$$

In [hardware/rtl/components/requantize.sv](../hardware/rtl/components/requantize.sv):

1. The INT32 accumulator is multiplied by `MULTIPLIER`.
2. The product uses a 64-bit temporary value.
3. An arithmetic right shift by `SHIFT` divides by $2^{SHIFT}$ while preserving the sign.
4. The result is saturated to INT8.

Saturation is:

$$
output =
\begin{cases}
127 & y > 127 \\
-128 & y < -128 \\
y & otherwise
\end{cases}
$$

The generated [data/mem/tinynpu_params.vh](../data/mem/tinynpu_params.vh) contains layer-specific multiplier and shift parameters.

## 14. RTL Components

### 14.1 MAC Unit

[hardware/rtl/components/mac.sv](../hardware/rtl/components/mac.sv) computes:

$$
result = bias + \sum_{i=0}^{N-1} input_i weight_i
$$

The inputs and weights are signed INT8. Each product is held in INT16 before being added into an INT32 accumulator.

The MAC is combinational, so its output changes when any input, weight, or bias changes.

### 14.2 Requantization Unit

[hardware/rtl/components/requantize.sv](../hardware/rtl/components/requantize.sv) converts INT32 to INT8 using the fixed-point operation described above.

### 14.3 ReLU Unit

[hardware/rtl/components/relu.sv](../hardware/rtl/components/relu.sv) computes:

$$
ReLU(x) = max(0,x)
$$

Negative INT8 values become zero. Nonnegative values are unchanged.

### 14.4 Neuron

[hardware/rtl/layers/neuron.sv](../hardware/rtl/layers/neuron.sv) combines:

```text
packed INT8 inputs
        |
        v
packed INT8 weights
        |
        v
INT32 MAC plus INT32 bias
        |
        v
INT32 x multiplier, arithmetic shift
        |
        v
INT8 saturation
        |
        v
ReLU
        |
        v
INT8 output
```

The current neuron uses packed buses:

```text
inputs_bus  : N x 8 bits
weights_bus : N x 8 bits
bias        : 32 bits
output      : 8 bits
```

### 14.5 Layers

The layer modules instantiate multiple neurons:

```text
layer1.sv: 21 inputs -> 14 neurons
layer2.sv: 14 inputs -> 7 neurons
layer3.sv: 7 inputs -> 1 neuron
```

Each layer supplies the appropriate multiplier, shift, packed weights, and packed biases.

### 14.6 Top-Level Network

[hardware/rtl/top/tinynpu.sv](../hardware/rtl/top/tinynpu.sv) connects the three layers:

```text
inputs_bus
    |
    v
layer1 -> layer1_output
    |
    v
layer2 -> layer2_output
    |
    v
layer3 -> output_value
```

The final RTL output is an INT8 activation, not an FP32 sigmoid probability.

## 15. Why the RTL Does Not Need Sigmoid

The Python model uses:

$$
p = sigmoid(z_3)$$

The classification threshold is $p >= 0.5$. Since:

$$
sigmoid(z_3) >= 0.5 \iff z_3 >= 0
$$

hardware can classify by checking whether the final signed activation is positive:

```text
final RTL activation > 0 -> class 1
final RTL activation <= 0 -> class 0
```

This avoids implementing an expensive exponential function in the small RTL design.

The Python quantized model still calculates sigmoid because it is used as a probability reference. The RTL only needs to preserve the class decision.

## 16. File Formats

### CSV

`*.csv` is human-readable tabular input data:

```text
feature_0,feature_1,...,feature_20,target
```

It is read by pandas.

### NPY

`*.npy` is NumPy's binary array format. It preserves:

- Array shape.
- Numeric dtype.
- Integer or floating-point values.

It is used for intermediate model artifacts such as:

```text
weights1_int8.npy
bias1_int32.npy
activation1_int8.npy
outputs_fp32.npy
```

### TXT

`*_scale.txt` stores one decimal scale value as text. Scales remain floating-point metadata and are easy to inspect or load from Python.

### MEM

`*.mem` contains one hexadecimal value per line for use with Verilog `$readmemh()`.

Examples:

```text
INT8 value  5   -> 05
INT8 value -1   -> FF
INT32 value 5   -> 00000005
INT32 value -1  -> FFFFFFFF
```

Negative values use two's-complement representation. For a signed value $v$ with $B$ bits, the stored bit pattern is:

$$
bit\_pattern = v \mod 2^B
$$

The exporter applies this with a mask:

$$
masked = v \ \& \ (2^B - 1)
$$

The `.mem` files contain flattened arrays, so the RTL testbench must pack values in the same order used by NumPy's `flatten()`.

### Verilog Header

`data/mem/tinynpu_params.vh` contains integer `localparam` values such as layer multipliers and shifts. It lets SystemVerilog testbenches use the same generated fixed-point parameters as the Python export.

### SystemVerilog

`*.sv` contains synthesizable RTL modules and simulation-only testbenches. The hardware modules are primarily combinational: there is no clocked pipeline in the current implementation.

## 17. Export Process

[tools/export_mem.py](../tools/export_mem.py) performs these conversions:

```text
inputs_int8.npy  -> inputs.mem
weights1_int8.npy -> weights1.mem
bias1_int32.npy   -> bias1.mem
weights2_int8.npy -> weights2.mem
bias2_int32.npy   -> bias2.mem
weights3_int8.npy -> weights3.mem
bias3_int32.npy   -> bias3.mem
targets_int8.npy  -> targets.mem
```

It also derives expected classes from the FP32 reference probabilities:

$$
expected\_class = (output\_fp32 >= 0.5)
$$

These are written to `expected_classes.mem` for RTL verification.

## 18. Verification Flow

[tools/verify_rtl.py](../tools/verify_rtl.py) performs four main jobs.

### Step 1: Load Memory Files

It reads the `.mem` files and converts hexadecimal two's-complement values back into signed integers.

For INT8:

$$
if\ value >= 0x80,\ value := value - 0x100
$$

For INT32:

$$
if\ value >= 0x80000000,\ value := value - 0x100000000
$$

### Step 2: Run a Bit-Accurate Python RTL Emulator

The emulator duplicates the hardware sequence:

```text
MAC plus bias
        |
        v
integer multiplier and right shift
        |
        v
INT8 saturation
        |
        v
ReLU
```

It compares the predicted class against `expected_classes.mem`.

### Step 3: Run Icarus Verilog

The script compiles:

```text
mac.sv
relu.sv
requantize.sv
neuron.sv
layer1.sv
layer2.sv
layer3.sv
tinynpu.sv
tb_tinynpu_real.sv
```

Then it runs the generated simulation with `vvp`.

### Step 4: Compare Results

The verification checks:

- Expected class versus RTL class.
- Python emulator result versus RTL result.
- Layer outputs and final outputs where provided by the testbench.

A successful run currently reports two samples, two classification matches, and zero bit mismatches.

## 19. Running the Project

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the full Python training, quantization, and export pipeline:

```powershell
python tools/main.py
```

Run quantized Python inference:

```powershell
python src/quantized_inference.py
```

Run real-model Python and RTL verification:

```powershell
python tools/verify_rtl.py
```

Compile and run one RTL testbench manually:

```powershell
$sources = Get-ChildItem hardware/rtl/components,hardware/rtl/layers,hardware/rtl/top -Filter *.sv
iverilog -g2012 -s tb_tinynpu -o hardware/sim/tb_tinynpu $sources hardware/rtl/testbenches/tb_tinynpu.sv
vvp hardware/sim/tb_tinynpu
```

## 20. Current Known Limitation

The current [tb_neuron.sv](../hardware/rtl/testbenches/tb_neuron.sv) is from the older array-style neuron interface. It refers to ports named `inputs` and `weights`, and internal signals named `mac_result` and `requantized_value`.

The current [neuron.sv](../hardware/rtl/layers/neuron.sv) uses packed ports named `inputs_bus` and `weights_bus`, and calculates its internal values directly. Therefore `tb_neuron.sv` does not compile, even though the neuron works inside the layer and full-network tests.

Updating that testbench to the current packed-bus interface is the next cleanup task.

## 21. Summary

TinyNPU follows this numerical transformation:

```text
FP32 CSV values
    |
    v
FP32 neural-network training
    |
    v
FP32 weights and biases
    |
    v
Symmetric INT8 weights, inputs, and activations
INT32 biases and accumulators
    |
    v
Fixed-point multiplier and right-shift requantization
    |
    v
INT8 ReLU layers in SystemVerilog
    |
    v
Final integer class decision
```

The central formulas are:

$$
z = Wx + b$$

$$
ReLU(z) = max(0,z)
$$

$$
sigmoid(z) = \frac{1}{1+e^{-z}}
$$

$$
q = clip\left(round\left(\frac{r}{scale}\right), -128, 127\right)
$$

$$
bias_{int32} = round\left(\frac{bias_{float}}{input\_scale \times weight\_scale}\right)
$$

$$
output_{int8} \approx (accumulator \times multiplier) >>> shift
$$

These formulas connect the Python reference model, the quantized model, the exported memory files, and the SystemVerilog hardware implementation.
