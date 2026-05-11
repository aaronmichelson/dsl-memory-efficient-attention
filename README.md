# dsl-memory-efficient-attention

Domain-Specific Language for Memory-Efficient Attention  
COMPSCI 790 — Domain Specific Programming for AI Applications  
University of Wisconsin–Milwaukee  
Aaron Michelson

---

## Overview

This project explores compiler-inspired and domain-specific programming techniques for improving the efficiency of transformer attention mechanisms, with a particular focus on memory-efficient attention similar to FlashAttention-style approaches.

The project was developed as part of the COMPSCI 790 graduate term project at the University of Wisconsin–Milwaukee. The work investigates how ideas from domain-specific languages (DSLs), operator fusion, JIT compilation, and tiled computation can reduce memory overhead and improve execution efficiency for large-scale transformer workloads.

The repository includes weekly experimental notebooks, benchmarking outputs, generated figures, and final evaluation artifacts used throughout the research and development process.

---

## Project Goals

The primary goals of this project are:

- Explore memory-efficient attention implementations inspired by FlashAttention
- Investigate compiler-style transformations for tensor operations
- Analyze GPU execution behavior and memory traffic
- Study the effects of kernel fusion and tiled computation
- Benchmark optimized attention workflows against naïve implementations
- Examine application scenarios where customized optimization provides tangible benefits

---

## Research Focus

This project emphasizes several major concepts in modern AI systems and compiler design:

- Domain-Specific Languages (DSLs)
- Memory-efficient attention mechanisms
- FlashAttention-style tiling strategies
- Operator fusion
- JIT compilation
- GPU kernel optimization
- CUDA execution efficiency
- Transformer scaling challenges
- Long-context inference optimization

---

## Repository Structure

```text
dsl-memory-efficient-attention/
│
├── figures/
│   ├── Generated visualizations and diagrams
│   ├── Performance charts
│   └── Experimental figures used in reports
│
├── notebooks/
│   ├── Week 1 notebooks
│   ├── Week 2 notebooks
│   ├── Week 3 notebooks
│   ├── Week 4 notebooks
│   ├── Week 5 notebooks
│   └── Week 6 notebooks
│
├── results/
│   ├── CSV benchmark outputs
│   ├── Profiling summaries
│   ├── Timing results
│   └── Experimental evaluation data
│
├── README.md
└── .gitignore
```

---

## Experimental Topics

The notebooks and experiments in this repository include:

- Baseline transformer attention implementations
- Memory-efficient attention variants
- Attention tiling strategies
- Profiling GPU memory usage
- Kernel launch overhead analysis
- JIT compilation experiments
- Tensor operation fusion
- Runtime benchmarking
- Throughput and latency comparisons
- Long-context attention analysis

---

## Technologies Used

### Programming Languages

- Python

### Frameworks and Libraries

- PyTorch
- JAX
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### Development Tools

- Google Colab
- Visual Studio Code
- GitHub
- CUDA-enabled GPU environments

---

## Hardware Environment

Experiments were conducted using NVIDIA GPU acceleration with CUDA-enabled environments for profiling and benchmarking attention workloads.

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

## Future Work

Potential future extensions of this project include:

- Triton-based kernel implementations
- Custom CUDA attention kernels
- Auto-tuning tile sizes
- Streaming attention architectures
- Multi-tenant inference optimization
- Long-context transformer benchmarking
- DSL abstractions for transformer compilation pipelines

---

## License

This repository is intended for academic and research purposes.
