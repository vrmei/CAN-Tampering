# CAN-Tampering

This repository contains tools and code related to **CAN Bus Fuzzing**, and the **MIDS model** used in the research paper. The project is structured into two main modules: **MIDS**, and **CANFUZZ**.

## Modules Overview

### MIDS
This module represents the primary contribution of the author's paper, and it includes the **MIDS model architecture**, **dataset construction scripts**, and **experimental data** along with **model weights**.

- **Functionality**:
  - Includes the model architecture as described in the paper.
  - Contains scripts for constructing datasets required for experiments.
  - Includes all experimental data recorded during the research process, as well as the trained model weights.
  
- **Status**: This is the core component of the paper, containing the full experimental workflow, data, and model training results.

### CANFUZZ
The **CANFUZZ** module contains a **CAN Bus Fuzzing** tool based on **PeakCAN**. It is still under development.

- **Functionality**: Used for fuzz testing the CAN bus by sending malformed packets to test the robustness of the system.
- **Status**: Currently in development; some features may be unstable.
