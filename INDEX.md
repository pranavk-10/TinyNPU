# TinyNPU - Complete Documentation Index

Welcome to TinyNPU! This file helps you navigate all the documentation.

## 📖 Documentation Files

### Quick Start (Start Here!)
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute quick reference guide
  - Project structure at a glance
  - Getting started in 3 steps
  - Common tasks and examples
  - Environment setup

### Main Documentation
- **[README.md](README.md)** - Comprehensive project documentation
  - Complete project overview
  - Network architecture details
  - Training and quantization explained
  - Hardware implementation guide
  - All technical details

### Organization & Structure
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed file organization
  - Complete directory tree
  - File organization principles
  - Data flow explanations
  - Running instructions
  - Version control guidelines

### Before & After
- **[BEFORE_AND_AFTER.md](BEFORE_AND_AFTER.md)** - What changed and why
  - Before: messy flat structure
  - After: organized hierarchy
  - Benefits comparison
  - Statistics

## 🗂️ Directory Guide

```
TinyNPU/
├── src/                 → Python ML algorithms
├── tools/               → Utilities & pipeline scripts
├── hardware/rtl/        → SystemVerilog RTL modules
├── data/                → Datasets and parameters
└── docs/                → Images and diagrams
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed descriptions.

## 🚀 Getting Started

### 1. Quick Start (5 minutes)
```bash
# Read quick reference
cat QUICKSTART.md

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python tools/main.py
```

### 2. Understand the Project (15 minutes)
- Read [README.md](README.md) - Full overview
- Read [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - How it's organized

### 3. Explore the Code
- **ML code**: Look in `src/`
- **Tools**: Look in `tools/`
- **Hardware**: Look in `hardware/rtl/`
- **Data**: Look in `data/`

## 🎯 By Use Case

### I want to...

#### Train the model
```bash
python tools/main.py
```
See [QUICKSTART.md](QUICKSTART.md#getting-started)

#### Understand the network architecture
See [README.md](README.md#network-architecture)

#### Find a specific file
See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md#directory-organization)

#### Modify the ML code
Look in `src/` - see [README.md](README.md#core-python-modules)

#### Modify the hardware
Look in `hardware/rtl/` - see [README.md](README.md#rtl-hardware-implementation)

#### Run simulations
See [QUICKSTART.md](QUICKSTART.md#simulate-full-network)

#### Understand quantization
See [README.md](README.md#quantization-strategy)

#### Add new files
See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md#adding-new-components)

#### Set up version control
See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md#version-control)

#### Understand what changed
See [BEFORE_AND_AFTER.md](BEFORE_AND_AFTER.md)

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| Python ML Modules | 5 |
| Utility Scripts | 3 |
| RTL Components | 3 |
| RTL Layers | 4 |
| Testbenches | 8 |
| Total Parameters | 421 |
| Network Layers | 3 (21→14→7→1) |

## 🔗 Quick Links

### Essential Files
- [README.md](README.md) - Main documentation
- [requirements.txt](requirements.txt) - Python dependencies
- [.gitignore](.gitignore) - Git ignore rules

### Key Directories
- [src/](src/) - Python ML code
- [tools/](tools/) - Pipeline utilities
- [hardware/rtl/](hardware/rtl/) - SystemVerilog modules
- [data/](data/) - Datasets and parameters

### Entry Points
- Pipeline: `python tools/main.py`
- Simulation: `cd hardware/rtl && iverilog ...`

## 📋 Reading Order

### For New Users
1. This file (you are here!)
2. [QUICKSTART.md](QUICKSTART.md) - 5 minutes
3. [README.md](README.md) - 15 minutes
4. Explore the code in `src/` and `hardware/rtl/`

### For Contributors
1. [QUICKSTART.md](QUICKSTART.md)
2. [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
3. [README.md](README.md) for technical details
4. [BEFORE_AND_AFTER.md](BEFORE_AND_AFTER.md) to understand changes

### For Team Leads
1. [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
2. [BEFORE_AND_AFTER.md](BEFORE_AND_AFTER.md)
3. [README.md](README.md)
4. Review code organization in `src/`, `tools/`, `hardware/`

## 🎓 Learning Path

### Understanding the Project
1. Learn about [Network Architecture](README.md#network-architecture)
2. Learn about the [Complete Pipeline](README.md#complete-pipeline)
3. Study [Training Details](README.md#training-details)
4. Study [Quantization Strategy](README.md#quantization-strategy)
5. Review [Hardware Implementation](README.md#rtl-hardware-implementation)

### Hands-On
1. Run `python tools/main.py` to train and quantize
2. Explore output in `data/`
3. Look at `.mem` files in `data/mem/`
4. Simulate hardware in `hardware/rtl/testbenches/`

### Advanced
1. Modify network architecture in `src/model.py`
2. Change quantization in `src/quantize_model.py`
3. Optimize RTL in `hardware/rtl/`
4. Add new components

## ⚡ Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python tools/main.py

# Simulate MAC unit
cd hardware/rtl
iverilog -g2012 -o ../sim/mac_sim components/mac.sv testbenches/tb_mac.sv
../sim/mac_sim

# Simulate full network
iverilog -g2012 -o ../sim/tinynpu_sim \
  components/*.sv layers/*.sv top/tinynpu.sv testbenches/tb_tinynpu.sv
../sim/tinynpu_sim

# Git commit
git add .
git commit -m "Your commit message"
```

## 🆘 Need Help?

1. **Getting started?** → Read [QUICKSTART.md](QUICKSTART.md)
2. **Finding files?** → Check [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
3. **Understanding code?** → See [README.md](README.md)
4. **What changed?** → Read [BEFORE_AND_AFTER.md](BEFORE_AND_AFTER.md)
5. **Specific topic?** → Use Ctrl+F to search in README.md

## 📚 Document Sizes

| Document | Size | Read Time |
|----------|------|-----------|
| QUICKSTART.md | 6.5 KB | 5 min |
| PROJECT_STRUCTURE.md | 8.1 KB | 10 min |
| README.md | 28 KB | 15 min |
| BEFORE_AND_AFTER.md | 7.6 KB | 10 min |
| This file | - | 2 min |

**Total documentation: ~50 KB, ~40 minutes to read everything**

## 🎯 Next Steps

1. **Read [QUICKSTART.md](QUICKSTART.md)** to understand the project layout
2. **Run `python tools/main.py`** to train the model
3. **Explore the generated files** in `data/`
4. **Read [README.md](README.md)** for full technical details
5. **Start contributing!**

---

**Welcome to TinyNPU!** 🚀

This is a comprehensive project demonstrating the complete journey from **Machine Learning → Quantization → Hardware Implementation → FPGA**.

All documentation is here to help you get started, understand the code, and contribute effectively.

Happy exploring!
