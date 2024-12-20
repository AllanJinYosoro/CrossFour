import re
import matplotlib.pyplot as plt
import os
import math

def parse_log_file(file_path):
    """
    解析日志文件，提取 Episode 和 Loss 信息，以及对手模型更新的时间点。
    
    Args:
        file_path (str): 日志文件路径。
    
    Returns:
        list: 包含 Episode 的列表。
        list: 包含对应 Loss 的列表。
        list: 对手模型更新的 Episode 列表。
    """
    episodes = []
    losses = []
    opponent_update_episodes = []

    # 正则表达式匹配日志中的 Episode 和 Loss 信息
    loss_pattern = r"Episode (\d+)/\d+, Total Reward: [^,]+, Loss: ([\d\.]+)"
    # 正则表达式匹配对手模型更新的日志信息
    opponent_update_pattern = r"Opponent model updated at episode (\d+)\."

    # 读取日志文件并解析
    with open(file_path, "r") as file:
        for line in file:
            # 匹配 Loss 信息
            loss_match = re.search(loss_pattern, line)
            if loss_match:
                episodes.append(int(loss_match.group(1)))
                losses.append(float(loss_match.group(2)))
            
            # 匹配对手模型更新的信息
            opponent_update_match = re.search(opponent_update_pattern, line)
            if opponent_update_match:
                opponent_update_episodes.append(int(opponent_update_match.group(1)))

    return episodes, losses, opponent_update_episodes


def plot_loss_curve(episodes, losses, opponent_update_episodes, output_path):
    """
    绘制 Loss 曲线并标记对手模型更新点，用竖虚线表示。
    
    Args:
        episodes (list): Episode 列表。
        losses (list): Loss 列表。
        opponent_update_episodes (list): 对手模型更新的 Episode 列表。
        output_path (str): 保存图像的路径。
    """
    # 创建一个画布，并设置大小
    plt.figure(figsize=(12, 6))

    # 第一个子图：普通Loss曲线
    plt.subplot(2, 1, 1)  # 参数分别为行数、列数、子图索引
    plt.plot(episodes, losses, label="Loss", color="blue", linewidth=1.5)
    for update_episode in opponent_update_episodes:
        plt.axvline(x=update_episode, color="red", linestyle="--", alpha=0.7, label="Opponent Model Updated" if update_episode == opponent_update_episodes[0] else "")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.legend()

    # 第二个子图：Log Loss曲线
    plt.subplot(2, 1, 2)
    plt.plot(episodes, [math.log(loss + 1e-9) for loss in losses], label="Log Loss", color="green", linewidth=1.5)
    for update_episode in opponent_update_episodes:
        plt.axvline(x=update_episode, color="red", linestyle="--", alpha=0.7, label="Opponent Model Updated" if update_episode == opponent_update_episodes[0] else "")
    plt.xlabel("Episode")
    plt.ylabel("Log Loss")
    plt.title("Log Loss Curve")
    plt.grid(True)
    plt.legend()

    # 调整子图间距
    plt.tight_layout()
    # 保存图像
    plt.savefig(output_path)
    print(f"Loss curves saved to {output_path}")
    # 关闭画布
    plt.close()

def get_latest_log_file(log_dir):
    """
    获取指定文件夹中按文件名排序最新的日志文件。
    
    Args:
        log_dir (str): 日志文件夹路径。
    
    Returns:
        str: 最新日志文件的路径。如果文件夹为空，则返回 None。
    """
    # 获取文件夹内所有文件的完整路径
    log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("train_log") and f.endswith(".txt")]
    
    if not log_files:
        return None  # 文件夹为空
    
    # 按文件名排序并返回最新的文件
    latest_file = sorted(log_files, key=lambda x: os.path.basename(x))[-1]
    return latest_file

if __name__ == "__main__":
    # 日志文件路径
    log_file = ''
    log_dir = "./log"
    # log_file = "log/train_log_20241220_145403.txt"
    
    if not os.path.isfile(log_file):  # 如果未指定文件或文件不存在
        log_file = get_latest_log_file(log_dir)
        if not log_file:
            print("No log files found in the specified directory.")
            exit(1)
        print(f"Using the latest log file: {log_file}")
    output_png = "./log/loss_curve.png"

    # 解析日志文件
    episodes, losses, opponent_update_episodes = parse_log_file(log_file)

    if episodes and losses:
        # 绘制并保存 Loss 曲线
        plot_loss_curve(episodes, losses, opponent_update_episodes, output_png)
    else:
        print("No valid data found in the log file.")