# Repository Name

This repository contains tools and code related to **CAN Bus Fuzzing**, **DBC File Storage**, and the **MIDS model** used in the research paper. The project is structured into three main modules: **MIDS**, **CANFUZZ**, and **DBC**.

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

### DBC
The **DBC** module stores a set of **DBC files** used for verifying the correctness of experiments. These files define the message formats and signal mappings used in CAN bus communication.

- **Functionality**: Stores and manages DBC files used to validate experimental correctness.
- **Use**: These files will serve as references during the validation process to ensure the accuracy of the data.
