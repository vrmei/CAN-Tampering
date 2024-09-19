import pandas as pd
import numpy as np
import argparse
import os
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

class DataModifier:
    def __init__(self, input_file, output_file, x, modify_type='CANID', seed=42):
        """
        初始化DataModifier类。

        Parameters:
        - input_file (str): 输入CSV文件路径。
        - output_file (str): 输出CSV文件路径。
        - x (int): 每隔x个报文修改一个报文。
        - modify_type (str): 修改类型，'CANID'、'payload'或'Both'。
        - seed (int): 随机种子，默认为42。
        """
        self.input_file = input_file
        self.output_file = output_file
        self.x = x
        self.modify_type = modify_type
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.data = None
        self.canid_col = 0
        self.payload_cols = list(range(1, 9))
        self.label_col = 9
        self.unique_canids = []
        self.unique_payloads = {}
        self.statistics = {
            'CANID': Counter(),
            'payload': defaultdict(Counter),
            'Total_Modifications': 0
        }
        # 用于存储修改前后的分布
        self.distribution_before = {}
        self.distribution_after = {}

    def read_data(self):
        """读取CSV数据集并初始化唯一值集合。"""
        try:
            self.data = pd.read_csv(self.input_file, delimiter=' ', header=None)
        except Exception as e:
            print(f"读取文件时出错: {e}")
            raise

        # 确认数据集至少有10列（CANID + 8个payload + label）
        print(self.data.shape)
        if self.data.shape[1] < 10:
            raise ValueError("数据集的列数少于10列，请检查数据格式。")

        # 获取所有唯一的CANID
        self.unique_canids = self.data[self.canid_col].unique()
        if len(self.unique_canids) == 0:
            raise ValueError("数据集中没有CANID，请检查数据。")

        # 获取每个payload列的唯一值
        for col in self.payload_cols:
            self.unique_payloads[col] = self.data[col].unique()

        print(f"共有 {len(self.unique_canids)} 个唯一的CANID。")
        for col in self.payload_cols:
            print(f"Payload列 {col} 有 {len(self.unique_payloads[col])} 个唯一值。")

    def capture_distribution(self, before=True):
        """
        捕获当前数据集的CANID和payload分布。

        Parameters:
        - before (bool): 如果为True，捕获修改前的分布；否则，捕获修改后的分布。
        """
        prefix = 'before' if before else 'after'
        self.distribution_before = {} if before else self.distribution_before
        self.distribution_after = {} if not before else self.distribution_after

        # CANID分布
        canid_counts = self.data[self.canid_col].value_counts().sort_index()
        if before:
            self.distribution_before['CANID'] = canid_counts
        else:
            self.distribution_after['CANID'] = canid_counts

        # payload分布
        for col in self.payload_cols:
            payload_counts = self.data[col].value_counts().sort_index()
            if before:
                self.distribution_before[f'Payload_{col}'] = payload_counts
            else:
                self.distribution_after[f'Payload_{col}'] = payload_counts

    def plot_distribution(self, before=True, save_dir='plots'):
        """
        生成并保存CANID和payload的分布图。

        Parameters:
        - before (bool): 如果为True，生成修改前的分布图；否则，生成修改后的分布图。
        - save_dir (str): 保存图像的目录。
        """
        prefix = 'before' if before else 'after'
        distribution = self.distribution_before if before else self.distribution_after

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for key, counts in distribution.items():
            plt.figure(figsize=(10, 6))
            sns.barplot(x=counts.index, y=counts.values, palette='viridis')
            plt.title(f'{prefix.capitalize()} Modification - {key} Distribution')
            plt.xlabel(key)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_filename = f"{prefix}_{key}_distribution.png"
            plt.savefig(os.path.join(save_dir, plot_filename))
            plt.close()
            print(f"保存分布图: {os.path.join(save_dir, plot_filename)}")

    def modify_canid(self, modify_idx):
        """修改指定索引的CANID，并记录统计信息。"""
        current_canid = self.data.at[modify_idx, self.canid_col]
        new_canid = random.choice(self.unique_canids)
        
        # 确保新的CANID与当前不同
        if new_canid == current_canid:
            possible_canids = self.unique_canids[self.unique_canids != current_canid]
            if len(possible_canids) > 0:
                new_canid = random.choice(possible_canids)
            else:
                print(f"只有一个唯一的CANID: {current_canid}, 无需修改。")
                return

        # 修改CANID
        self.data.at[modify_idx, self.canid_col] = new_canid
        # 设置标签为1
        self.data.at[modify_idx, self.label_col] = 1

        # 记录统计
        self.statistics['CANID'][new_canid] += 1
        self.statistics['Total_Modifications'] += 1

        print(f"修改CANID - 报文索引: {modify_idx}, 原CANID: {current_canid}, 新CANID: {new_canid}, 标签设置为1.")

    def modify_payload(self, modify_idx):
        """修改指定索引的一个payload值，并记录统计信息。"""
        # 随机选择一个payload列
        payload_to_modify = random.choice(self.payload_cols)
        current_payload = self.data.at[modify_idx, payload_to_modify]
        new_payload = random.choice(self.unique_payloads[payload_to_modify])

        # 确保新的payload值与当前不同
        if new_payload == current_payload:
            possible_payloads = self.unique_payloads[payload_to_modify][self.unique_payloads[payload_to_modify] != current_payload]
            if len(possible_payloads) > 0:
                new_payload = random.choice(possible_payloads)
            else:
                print(f"Payload列 {payload_to_modify} 只有一个唯一值: {current_payload}, 无需修改。")
                return

        # 修改payload
        self.data.at[modify_idx, payload_to_modify] = new_payload
        # 设置标签为1
        self.data.at[modify_idx, self.label_col] = 1

        # 记录统计
        self.statistics['payload'][payload_to_modify][new_payload] += 1
        self.statistics['Total_Modifications'] += 1

        print(f"修改payload - 报文索引: {modify_idx}, Payload列: {payload_to_modify}, 原值: {current_payload}, 新值: {new_payload}, 标签设置为1.")

    def modify_data(self):
        """执行数据集的修改操作。"""
        total_messages = len(self.data)
        num_modifications = total_messages // self.x

        print(f"总报文数: {total_messages}, 每隔 {self.x} 个报文修改一个, 需要修改的报文数: {num_modifications}")
        print(f"修改类型: {self.modify_type}")

        for i in range(num_modifications):
            start_idx = i * self.x
            end_idx = start_idx + self.x
            if end_idx > total_messages:
                end_idx = total_messages

            window = self.data.iloc[start_idx:end_idx]
            modify_idx = window.sample(n=1, random_state=self.seed + i).index[0]

            if self.modify_type == 'CANID':
                self.modify_canid(modify_idx)
            elif self.modify_type == 'payload':
                self.modify_payload(modify_idx)
            elif self.modify_type == 'Both':
                # 随机选择修改CANID或payload
                choice = random.choice(['CANID', 'payload'])
                if choice == 'CANID':
                    self.modify_canid(modify_idx)
                else:
                    self.modify_payload(modify_idx)
            else:
                raise ValueError(f"未知的修改类型: {self.modify_type}")

    def save_data(self):
        """保存修改后的数据集。"""
        try:
            self.data.to_csv(self.output_file, header=False, index=False)
            print(f"修改后的数据已保存到 {self.output_file}")
        except Exception as e:
            print(f"保存文件时出错: {e}")
            raise

    def log_statistics(self, log_file="modification_statistics.txt"):
        """记录修改统计信息到日志文件。"""
        try:
            with open(log_file, "a") as f:
                f.write(f"\nModification Statistics for {self.output_file}\n")
                f.write(f"Modify Type: {self.modify_type}\n")
                f.write(f"Total Modifications: {self.statistics['Total_Modifications']}\n\n")
                
                if self.modify_type in ['CANID', 'Both']:
                    f.write("CANID Modifications:\n")
                    for canid, count in self.statistics['CANID'].items():
                        f.write(f"  CANID {canid}: {count} times\n")
                    f.write("\n")
                
                if self.modify_type in ['payload', 'Both']:
                    f.write("Payload Modifications:\n")
                    for payload_col, changes in self.statistics['payload'].items():
                        f.write(f"  Payload Column {payload_col}:\n")
                        for value, count in changes.items():
                            f.write(f"    Value {value}: {count} times\n")
                    f.write("\n")
            print(f"修改统计信息已记录到 {log_file}")
        except Exception as e:
            print(f"记录统计信息时出错: {e}")
            raise

    def plot_distributions_before_after(self, save_dir='plots'):
        """生成并保存修改前后的分布图。"""
        self.plot_distribution(before=True, save_dir=save_dir)
        self.capture_distribution(before=False)
        self.plot_distribution(before=False, save_dir=save_dir)

    def capture_and_plot_distributions(self, save_dir='plots'):
        """捕获并绘制修改前后的分布图。"""
        print("捕获修改前的分布...")
        self.capture_distribution(before=True)
        self.plot_distribution(before=True, save_dir=save_dir)

        print("开始修改数据...")
        self.modify_data()

        print("捕获修改后的分布...")
        self.capture_distribution(before=False)
        self.plot_distribution(before=False, save_dir=save_dir)

    def generate_plots(self, save_dir='plots'):
        """生成修改前后的分布图。"""
        # 修改前已经被捕获并绘制
        # 修改后需要再次捕获和绘制
        self.plot_distributions_before_after(save_dir=save_dir)

def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="修改数据集中的CANID或payload，并设置标签。")
    parser.add_argument("--input_file", type=str, default='data/owndata/processed/white.csv',required=True, help="输入CSV文件路径。")
    parser.add_argument("--output_file", type=str, default='data/owndata/attackdata', required=True, help="输出CSV文件路径。")
    parser.add_argument("--x", type=int, default=100, required=True, help="每隔x个报文修改一个报文。")
    parser.add_argument("--modify_type", type=str, choices=['CANID', 'payload', 'Both'], default='CANID', help="修改类型: 'CANID'、'payload' 或 'Both'。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认值为42。")
    parser.add_argument("--log_file", type=str, default="modification_statistics.txt", help="统计信息日志文件路径。")
    parser.add_argument("--save_dir", type=str, default="plots", help="分布图保存目录。")
    return parser.parse_args()

def main():
    args = parse_arguments()

    if not os.path.isfile(args.input_file):
        print(f"输入文件不存在: {args.input_file}")
        return

    modifier = DataModifier(
        input_file=args.input_file,
        output_file=args.output_file,
        x=args.x,
        modify_type=args.modify_type,
        seed=args.seed
    )

    print("开始读取数据...")
    modifier.read_data()

    print("捕获并绘制修改前的分布图...")
    modifier.capture_distribution(before=True)
    modifier.plot_distribution(before=True, save_dir=args.save_dir)

    print("开始修改数据...")
    modifier.modify_data()

    print("捕获并绘制修改后的分布图...")
    modifier.capture_distribution(before=False)
    modifier.plot_distribution(before=False, save_dir=args.save_dir)

    print("保存修改后的数据...")
    modifier.save_data()

    print("记录统计信息...")
    modifier.log_statistics(args.log_file)

    print("生成并保存修改前后的分布图...")
    modifier.generate_plots(save_dir=args.save_dir)

    print("数据修改和统计完成。")

if __name__ == "__main__":
    main()
