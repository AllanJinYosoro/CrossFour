import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import argparse
from datetime import datetime
from assets.connectfour import ConnectFourEnv
import warnings
import logging
import os
from zoo.DQN import DQN
from zoo.Rainbow import RainbowDQN
from util.ReplayBuffer import ReplayBuffer
from util.PRBuffer import PrioritizedReplayBuffer

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建日志文件夹并配置日志记录
log_filename = f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_dir = "./log"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, log_filename),
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
logging.info("Training started.")

# 获取动作函数
def get_action_from_atoms(state, q_network):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # 添加批量维度
    with torch.no_grad():
        q_atoms = q_network(state_tensor)
        q_values = q_network.get_q_values(q_atoms)
    return torch.argmax(q_values).item()


def get_action_from_Q(state, q_network):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # 添加批量维度
    with torch.no_grad():
        q_values = q_network(state_tensor)  # 获取所有动作的 Q 值
    return torch.argmax(q_values).item()  # 返回 Q 值最高的动作


# 计算目标分布
def compute_target_distribution(rewards, dones, next_q_atoms, gamma, support, atom_size, v_min, v_max):
    delta_z = (v_max - v_min) / (atom_size - 1)
    batch_size = rewards.size(0)

    # 计算目标分布的投影
    target_support = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma * support.unsqueeze(0)
    target_support = target_support.clamp(v_min, v_max)

    b = (target_support - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    offset = torch.linspace(0, (batch_size - 1) * atom_size, batch_size).long().unsqueeze(1).to(rewards.device)

    m = torch.zeros(batch_size, atom_size).to(rewards.device)

    # 处理特殊情况：b 恰好为整数时
    exact_match = (b == l) & (b == u)  # 确定 b 为整数的位置
    if exact_match.any():
        exact_indices = (l + offset)[exact_match]  # 计算整数匹配的平面索引
        m.view(-1).index_add_(0, exact_indices.view(-1), next_q_atoms[exact_match].view(-1))
    m.view(-1).index_add_(0, (l + offset).view(-1), (next_q_atoms * (u - b)).view(-1))
    m.view(-1).index_add_(0, (u + offset).view(-1), (next_q_atoms * (b - l)).view(-1))

    return m


# 训练函数
def train(args):
    """
    使用 DQN 训练智能体。
    Args:
        args: 包含超参数和文件路径的命令行参数。
    """
    # 超参数
    learning_rate = args.learning_rate
    gamma = args.gamma
    batch_size = args.batch_size
    buffer_capacity = args.buffer_capacity
    target_update_frequency = args.target_update_frequency
    num_episodes = args.num_episodes
    atom_size = args.atom_size  # 分类分布中的原子数量
    alpha = args.alpha
    v_min = -1  # 支持的最小值
    v_max = 1   # 支持的最大值

    # 初始化环境、Q 网络、目标网络、优化器和经验回放缓冲区
    env = ConnectFourEnv(rows=6, cols=7, win_condition="four_in_a_row")
    input_dim = (env.rows, env.cols)  # 展平后的棋盘大小
    output_dim = env.cols  # 动作数量（列数）

    q_network = RainbowDQN(input_dim, output_dim, atom_size=atom_size, v_min=v_min, v_max=v_max).to(device)
    target_network = RainbowDQN(input_dim, output_dim, atom_size=atom_size, v_min=v_min, v_max=v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    # 尝试加载保存的模型
    try:
        q_network.load_state_dict(torch.load(args.cont_model_path))
        target_network.load_state_dict(q_network.state_dict())
        logging.info("Loaded saved model for continued training.")
    except FileNotFoundError:
        logging.info("No saved model found. Starting training from scratch.")

    # 加载对手模型（预训练模型）
    opponent_model = RainbowDQN(input_dim, output_dim).to(device)
    try:
        opponent_model.load_state_dict(torch.load(args.oppo_model_path))
        opponent_model.eval()  # 确保对手模型固定（不更新）
        logging.info("Loaded opponent model.")
    except FileNotFoundError:
        logging.info("No opponent model found. Using a random opponent.")

    # 初始化经验回放缓冲区
    replay_buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=alpha)

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        episode_transitions = []  # 存储智能体的转移
        total_reward = 0  # 记录每个 episode 的总奖励

        ############## 智能体与对手的游戏循环 ##############
        while not done:
            # 智能体的动作（探索与利用）
            action = get_action_from_atoms(state, q_network)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward  # 累积奖励

            if done:
                episode_transitions.append((state, action, reward, next_state, done))
                break

            # 对手使用最大 Q 策略
            opponent_action = get_action_from_atoms(next_state, opponent_model)
            state_after_opponent, opponent_reward, done, _ = env.step(opponent_action)

            # 存储智能体的转移
            episode_transitions.append((state, action, reward, state_after_opponent, done))

            # 更新当前状态
            state = state_after_opponent.copy()

        # 将智能体的转移存入经验回放缓冲区
        for transition in episode_transitions:
            replay_buffer.push(*transition)

        ############## 执行经验回放并更新 Q 网络 ##############
        if len(replay_buffer) >= batch_size:
            beta = min(1.0, args.beta_start + episode * (1.0 - args.beta_start) / args.num_episodes)

            states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta=beta)
            states = torch.FloatTensor(states).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).unsqueeze(1).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            dones = torch.FloatTensor(dones).to(device)

            # 重置噪声
            q_network.reset_noise()
            target_network.reset_noise()

            # 计算目标分布
            next_q_atoms = q_network(next_states)
            next_q_values = q_network.get_q_values(next_q_atoms)
            next_actions = next_q_values.argmax(1).unsqueeze(1)

            next_q_atoms = target_network(next_states)
            next_q_atoms = next_q_atoms.gather(1, next_actions.unsqueeze(-1).expand(-1, -1, atom_size)).squeeze(1)

            target_distribution = compute_target_distribution(
                rewards, dones, next_q_atoms, gamma, q_network.support, atom_size, v_min, v_max
            )

            # 计算当前 Q 值分布
            q_atoms = q_network(states)
            q_atoms = q_atoms.gather(1, actions.unsqueeze(-1).expand(-1, -1, atom_size)).squeeze(1)

            # 损失计算：KL 散度
            weights = torch.FloatTensor(weights).to(device)
            loss = F.kl_div(q_atoms.log(), target_distribution, reduction="none")
            loss = (loss.sum(dim=1) * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算 TD 误差
            predicted_q_values = torch.sum(q_atoms * q_network.support, dim=1) 
            target_q_values = torch.sum(target_distribution * q_network.support, dim=1)
            td_errors = torch.abs(predicted_q_values - target_q_values).detach().cpu().numpy()

            priorities = td_errors + 1e-6  # 加入一个小正数避免优先级为 0
            replay_buffer.update_priorities(indices, priorities)

        # 更新目标网络
        if episode % target_update_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())
            torch.save(q_network.state_dict(), args.save_model_path)

        # 更新对手网络
        if episode % args.opponent_update_frequency == 0:
            opponent_model.load_state_dict(q_network.state_dict())
            logging.info(f"Opponent model updated at episode {episode}.")

        # 打印进度并记录日志
        if episode % 100 == 0:
            logging.info(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Loss: {loss.item():.8f}")
            print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Loss: {loss.item():.8f}")

    # 保存最终训练模型
    torch.save(q_network.state_dict(), args.save_model_path)
    logging.info("Training complete and model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent on Connect Four.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    parser.add_argument("--atom_size", type=int, default=51, help="Atom size for distributional DQN.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Factor for PER, greater alpha more based on priorities.")
    parser.add_argument("--beta_start", type=float, default=0.8, help="Beta factor for PER, gradually grows to 1.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for experience replay.")
    parser.add_argument("--buffer_capacity", type=int, default=100000, help="Replay buffer capacity.")
    parser.add_argument("--target_update_frequency", type=int, default=1000, help="Frequency of updating target network.")
    parser.add_argument("--opponent_update_frequency", type=int, default=10000, help="Frequency of updating opponent network.")
    parser.add_argument("--num_episodes", type=int, default=2000000, help="Number of episodes for training.")
    parser.add_argument("--oppo_model_path", type=str, default="./model/Rainbow2.pth", help="Path to load the opponent model.")
    parser.add_argument("--cont_model_path", type=str, default="./model/Rainbow2.pth", help="Path to load the continue model.")
    parser.add_argument("--save_model_path", type=str, default="./model/Rainbow3.pth", help="Path to save the model.")
    args = parser.parse_args()

    train(args)