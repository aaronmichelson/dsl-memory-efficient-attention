# dsl-memory-efficient-attention

Domain-Specific Language for Memory-Efficient Attention  
COMPSCI 790 — Domain Specific Programming for AI Applications  
University of Wisconsin–Milwaukee  
Aaron Michelson

---

## Overview

This repository contains the implementation, experiments, benchmarking notebooks, figures, and evaluation outputs for a graduate research term project focused on memory-efficient transformer attention mechanisms.

The project investigates how compiler-inspired optimizations and domain-specific programming techniques can improve transformer attention performance while reducing GPU memory overhead. The work is heavily inspired by FlashAttention-style approaches, tiled computation strategies, operator fusion, and modern AI compiler systems.

The repository includes six weeks of progressively developed experiments, beginning with profiling and baseline analysis and culminating in final benchmarking and evaluation studies.

---

## Project Objectives

The goals of this project include:

- Explore memory-efficient transformer attention implementations
- Investigate FlashAttention-style tiled computation
- Analyze GPU memory behavior during attention execution
- Evaluate compiler-style operator fusion strategies
- Study JIT compilation and execution overhead
- Benchmark runtime and memory efficiency improvements
- Examine long-context transformer optimization techniques
- Develop reproducible experimental workflows for AI systems research

---

## Research Areas

This project focuses on several major topics in modern AI systems and compiler research:

- Domain-Specific Languages (DSLs)
- Memory-Efficient Attention
- FlashAttention-Inspired Optimization
- GPU Kernel Fusion
- JIT Compilation
- Transformer Systems
- CUDA Performance Analysis
- Tensor Computation Optimization
- Long-Context Inference
- AI Compiler Techniques

---

## Repository Structure

```text
dsl-memory-efficient-attention/
│
├── figures/
│   ├── Performance plots
│   ├── Experimental visualizations
│   ├── Attention benchmarking charts
│   └── Figures used in the final report
│
├── notebooks/
│   ├── Week 1 exploratory profiling
│   ├── Week 2 optimization experiments
│   ├── Week 3 JIT and fusion analysis
│   ├── Week 4 attention implementation studies
│   ├── Week 5 benchmarking experiments
│   └── Week 6 final evaluation notebooks
│
├── results/
│   ├── Benchmark CSV outputs
│   ├── Profiling summaries
│   ├── Runtime measurements
│   └── Experimental evaluation data
│
├── tables/
│   ├── Final report tables
│   ├── Experimental summary tables
│   └── Quantitative evaluation outputs
│
├── README.md
└── .gitignore
```

---

## Experimental Workflow

### Weeks 1–2
- Baseline transformer attention analysis
- GPU profiling exploration
- Initial benchmarking setup
- Runtime and memory inspection

### Weeks 3–4
- JIT compilation experiments
- Operator fusion analysis
- Attention optimization studies
- Intermediate implementation evaluation

### Weeks 5–6
- Final benchmarking and evaluation
- Comparative runtime analysis
- Memory efficiency measurements
- Quantitative CSV result generation
- Final report figures and tables

---

## Experimental Topics

The repository includes experiments involving:

- Naïve transformer attention
- Memory-efficient attention
- Tiled attention execution
- Attention kernel optimization
- GPU memory profiling
- Runtime benchmarking
- JIT compilation overhead
- Kernel fusion performance
- Long-context transformer scaling
- Throughput and latency analysis

---

## Technologies Used

### Programming Language
- Python

### Frameworks and Libraries
- PyTorch
- JAX
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### Development Environment
- Google Colab
- Visual Studio Code
- GitHub
- CUDA-enabled GPU systems

---

## Hardware and Execution Environment

Experiments were performed using NVIDIA CUDA-enabled GPU environments for transformer attention benchmarking and profiling analysis.

---

## Course Information

**Course:** COMPSCI 790 — Domain Specific Programming for AI Applications  
**Institution:** University of Wisconsin–Milwaukee  
**Semester:** Spring 2026

---

## Author

**Aaron Michelson**  
M.S. Computer Science  
University of Wisconsin–Milwaukee

---

## Future Directions

Potential future extensions of this work include:

- Triton-based attention kernels
- Auto-tuned tiling strategies
- Custom CUDA implementations
- Streaming transformer inference
- Multi-tenant LLM optimization
- Advanced long-context attention systems
- Compiler-assisted transformer execution pipelines

---

## Academic Purpose

This repository was created for graduate academic research and educational purposes as part of a term project in AI systems and domain-specific programming.
