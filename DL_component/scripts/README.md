# README

## English Version

### **Project Overview**

This project provides a Python script, `modify_data.py`, designed to modify a dataset of CAN (Controller Area Network) messages. The dataset consists of records with a `CANID`, eight `payload` values, and a `label`. The script allows users to:

- **Modify `CANID`:** Replace the `CANID` of selected messages with an existing `CANID` from the dataset.
- **Modify `payload`:** Change a randomly selected `payload` value to another existing value within the same column.
- **Set Labels:** After modification, set the label of the altered message to `1`.
- **Generate Statistical Reports:** Collect and log detailed statistics of the modifications.
- **Visualize Data Distributions:** Create distribution plots of `CANID` and `payload` values before and after modifications.

### **Features**

- **Modular Design:** The script is organized into classes and functions for better readability and maintenance.
- **Flexible Modification Options:** Choose to modify `CANID`, `payload`, or both.
- **Statistical Logging:** Detailed statistics of modifications are recorded in a log file.
- **Distribution Plots:** Visual representations of data distributions before and after modifications.
- **Random Seed Control:** Ensure reproducibility of results by setting a random seed.

### **Requirements**

- Python 3.6 or higher
- Python Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn

### **Installation**

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/modify-canid-payload.git
   cd modify-canid-payload
   ```

2. **Install Required Libraries:**

   It's recommended to use a virtual environment.

   ```bash
   pip install pandas numpy matplotlib seaborn
   ```

### **Usage**

Run the script using the command line with the desired arguments.

```bash
python scripts/modify_data.py --input_file path/to/your/input.csv --output_file path/to/your/output.csv --x 100 --modify_type payload
```

#### **Arguments:**

- `--input_file`: **(Required)** Path to the input CSV file.
- `--output_file`: **(Required)** Path where the modified CSV will be saved.
- `--x`: **(Required)** Interval for modifications. For example, `--x 100` means every 100 messages, one message will be modified.
- `--modify_type`: **(Optional)** Type of modification. Choices are:
  - `CANID`: Modify only the `CANID`.
  - `payload`: Modify only the `payload`.
  - `Both`: Modify both `CANID` and `payload`.
  
  Default is `CANID`.
- `--seed`: **(Optional)** Random seed for reproducibility. Default is `42`.
- `--log_file`: **(Optional)** Path to the log file where statistics will be recorded. Default is `modification_statistics.txt`.
- `--save_dir`: **(Optional)** Directory where distribution plots will be saved. Default is `plots`.

#### **Examples:**

- **Modify Only CANID:**

  ```bash
python scripts/modify_data.py --input_file data/owndata/processed/white.csv --output_file data/owndata/attackdata/1.csv --x 100 --modify_type CANID  ```

- **Modify Only Payload:**

  ```bash
  python modify_data.py --input_file data/original_data.csv --output_file data/modified_data.csv --x 100 --modify_type payload
  ```

- **Modify Both CANID and Payload:**

  ```bash
  python modify_data.py --input_file data/original_data.csv --output_file data/modified_data.csv --x 100 --modify_type Both
  ```

### **Output Files**

1. **Modified CSV File:**

   The script saves the modified dataset to the specified `--output_file`.

2. **Log File:**

   Detailed statistics of the modifications are appended to the specified `--log_file`. This includes:

   - Total number of modifications.
   - Breakdown of modifications by `CANID` and each `payload` column.
   
3. **Distribution Plots:**

   Visual distribution plots before and after modifications are saved in the specified `--save_dir` directory. File names follow the pattern:

   - `before_CANID_distribution.png`
   - `after_CANID_distribution.png`
   - `before_Payload_1_distribution.png`
   - `after_Payload_1_distribution.png`
   - ...

### **Statistics and Visualization**

After running the script, you will find comprehensive statistics in the log file and visual distribution plots in the `plots` directory.

- **Log File Example:**

  ```
  Modification Statistics for data/modified_data.csv
  Modify Type: payload
  Total Modifications: 100

  Payload Modifications:
    Payload Column 3:
      Value 200: 5 times
      Value 180: 3 times
      ...
    Payload Column 7:
      Value 50: 4 times
      Value 70: 6 times
      ...
  ```

- **Distribution Plots:**

  The plots provide a visual comparison of data distributions before and after modifications, helping you assess the impact of your data alterations.

### **Notes**

- **Data Backup:** Always backup your original data before performing modifications.
- **Data Format:** Ensure that the input CSV file has no headers and exactly 10 columns: `CANID`, eight `payload` columns, and `label`.
- **Label Modification:** The script sets the label of modified messages to `1`. Ensure this aligns with your data labeling conventions.
- **Performance:** For very large datasets, the script may take some time to process. Optimization or batch processing may be necessary.

### **Extending Functionality**

You can further enhance the script by:

- **Excluding Specific Values:** Modify the selection logic to exclude certain `CANID` or `payload` values from being selected as replacements.
- **Batch Processing Multiple Files:** Adapt the script to handle multiple input files in a single run.
- **Advanced Statistics:** Incorporate additional statistical measures or visualizations to gain deeper insights.
- **Parallel Processing:** Implement multi-threading or multi-processing for faster execution on large datasets.

### **Contact**

For any questions or suggestions, please contact [your.email@example.com](mailto:your.email@example.com).

---

## 中文版

### **项目概述**

本项目提供了一个Python脚本 `modify_data.py`，用于修改CAN（控制器局域网络）消息的数据集。数据集由包含 `CANID`、八个 `payload` 值和一个 `label` 的记录组成。该脚本允许用户：

- **修改 `CANID`：** 将选定消息的 `CANID` 替换为数据集中已存在的 `CANID`。
- **修改 `payload`：** 将随机选择的 `payload` 值更改为同一列中另一个已存在的值。
- **设置标签：** 修改后，将被更改消息的标签设置为 `1`。
- **生成统计报告：** 收集并记录修改的详细统计信息。
- **可视化数据分布：** 在修改前后生成并保存 `CANID` 和 `payload` 值的分布图，帮助直观了解数据变化。

### **功能**

- **模块化设计：** 脚本被组织成类和函数，提升了代码的可读性和可维护性。
- **灵活的修改选项：** 可选择只修改 `CANID`、只修改 `payload` 或同时修改两者。
- **统计日志记录：** 修改的详细统计信息记录在日志文件中。
- **分布图生成：** 在修改前后生成 `CANID` 和 `payload` 的分布图，帮助直观了解数据变化。
- **随机种子控制：** 通过设置随机种子确保结果的可复现性。

### **需求**

- Python 3.6 及以上版本
- Python 库：
  - pandas
  - numpy
  - matplotlib
  - seaborn

### **安装**

1. **克隆仓库：**

   ```bash
   git clone https://github.com/yourusername/modify-canid-payload.git
   cd modify-canid-payload
   ```

2. **安装所需库：**

   建议使用虚拟环境。

   ```bash
   pip install pandas numpy matplotlib seaborn
   ```

### **使用方法**

使用命令行运行脚本，并提供所需参数。

```bash
python modify_data.py --input_file path/to/your/input.csv --output_file path/to/your/output.csv --x 100 --modify_type payload
```

#### **参数说明：**

- `--input_file`：**（必需）**输入CSV文件的路径。
- `--output_file`：**（必需）**修改后的CSV文件保存路径。
- `--x`：**（必需）**修改的间隔。例如，`--x 100` 表示每100个报文修改一个。
- `--modify_type`：**（可选）**修改类型。选项包括：
  - `CANID`：只修改 `CANID`。
  - `payload`：只修改 `payload`。
  - `Both`：同时修改 `CANID` 和 `payload`。
  
  默认值为 `CANID`。
- `--seed`：**（可选）**随机种子，确保结果的可复现性。默认值为 `42`。
- `--log_file`：**（可选）**统计信息日志文件的路径。默认值为 `modification_statistics.txt`。
- `--save_dir`：**（可选）**分布图保存的目录。默认值为 `plots`。

#### **示例：**

- **只修改 CANID：**

  ```bash
  python modify_data.py --input_file data/original_data.csv --output_file data/modified_data.csv --x 100 --modify_type CANID
  ```

- **只修改 payload：**

  ```bash
  python modify_data.py --input_file data/original_data.csv --output_file data/modified_data.csv --x 100 --modify_type payload
  ```

- **同时修改 CANID 和 payload：**

  ```bash
  python modify_data.py --input_file data/original_data.csv --output_file data/modified_data.csv --x 100 --modify_type Both
  ```

### **输出文件**

1. **修改后的 CSV 文件：**

   脚本将修改后的数据保存到指定的 `--output_file` 路径。

2. **日志文件：**

   修改的详细统计信息记录在指定的 `--log_file` 文件中。内容包括：

   - 总修改次数。
   - 按 `CANID` 和每个 `payload` 列的修改次数细分。

3. **分布图：**

   修改前后 `CANID` 和 `payload` 列的分布图保存在指定的 `--save_dir` 目录中。文件名遵循以下模式：

   - `before_CANID_distribution.png`
   - `after_CANID_distribution.png`
   - `before_Payload_1_distribution.png`
   - `after_Payload_1_distribution.png`
   - ...

### **统计与可视化**

运行脚本后，您将在日志文件中找到详细的统计信息，并在 `plots` 目录中找到分布图。

- **日志文件示例：**

  ```
  Modification Statistics for data/modified_data.csv
  Modify Type: payload
  Total Modifications: 100

  Payload Modifications:
    Payload Column 3:
      Value 200: 5 times
      Value 180: 3 times
      ...
    Payload Column 7:
      Value 50: 4 times
      Value 70: 6 times
      ...
  ```

- **分布图：**

  分布图直观展示了数据修改前后 `CANID` 和 `payload` 列的分布变化，帮助您评估数据修改的影响。

### **注意事项**

- **数据备份：** 在运行脚本之前，建议备份原始数据，以防止数据丢失或错误修改。
- **数据格式：** 确保输入的CSV文件符合预期格式（无标题，10列，分别为 `CANID`、8个 `payload` 和 `label`）。
- **标签修改：** 脚本将被修改报文的标签设置为 `1`，请确保这符合您的数据标注需求。
- **性能考虑：** 对于非常大的数据集，脚本的运行时间可能较长。可以考虑优化或分批处理数据。

### **扩展功能**

您可以根据需要进一步扩展脚本，例如：

- **排除特定值：** 在选择新的 `CANID` 或 `payload` 值时排除特定的值。
- **批量处理多个文件：** 适应一次处理多个输入文件。
- **更多统计指标：** 增加额外的统计测量或可视化，深入分析数据变化。
- **并行处理：** 对于大规模数据集，可以考虑使用多线程或多进程来加速处理。

### **联系方式**

如有任何问题或建议，请联系 [2863829951@qq.com](mailto:2863829951@qq.com)。

---