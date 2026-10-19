"""Vòng huấn luyện PPO. Khung – nhóm hoàn thiện phần TODO."""
import torch


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    # Generalized Advantage Estimation
    adv, last = torch.zeros_like(rewards), 0.0
    for t in reversed(range(len(rewards))):
        next_v = 0.0 if dones[t] or t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * (0.0 if dones[t] else last)
        adv[t] = last
    return adv, adv + values


def ppo_loss(new_logp, old_logp, adv, returns, values, entropy, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    ratio = torch.exp(new_logp - old_logp)
    policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv).mean()
    value_loss = (returns - values).pow(2).mean()
    return policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()


def train(env_factory, encoder, head, cfg):
    # TODO:
    # 1. Mỗi vòng: sinh nhiều episode với (A, B, ngân sách, giờ xuất phát) ngẫu nhiên
    # 2. Lưu (obs, action, logp, reward, value, done)
    # 3. Tính GAE, chuẩn hoá advantage, cập nhật nhiều epoch theo minibatch với ppo_loss
    # 4. Định kỳ đánh giá bằng routing.evaluate và lưu checkpoint vào checkpoints/
    raise NotImplementedError
