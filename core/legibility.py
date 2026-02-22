import numpy as np
from config import BETA_RATIONALITY, EFFORT_WEIGHT


def compute_reward(action, state, goal):
    dist_before = np.linalg.norm(goal - state)
    if dist_before < 1e-3:
        return 0.0
    dist_after = np.linalg.norm(goal - (state + action))
    return (dist_before - dist_after) - EFFORT_WEIGHT * np.linalg.norm(action)


def pi_H(action, state, goal, beta=BETA_RATIONALITY):
    return np.exp(beta * compute_reward(action, state, goal))


def compute_legibility(action, state, goals, target_idx):
    """
    Compute action legibility without belief prior.
    Measures pure discriminative power of the action.
    
    L(a|s,θ*) = log π_H(a|s,θ*) - log max_{θ≠θ*} π_H(a|s,θ)
    """
    if np.linalg.norm(action) < 1e-3:
        return 0.0
    
    # 计算所有goal的likelihood（不乘belief）
    likelihoods = np.array([pi_H(action, state, g) for g in goals])
    
    # Target goal的likelihood
    target_likelihood = likelihoods[target_idx]
    
    # 最强竞争者的likelihood
    other_likelihoods = [likelihoods[i] for i in range(len(goals)) if i != target_idx]
    max_other = max(other_likelihoods) if other_likelihoods else 1e-10
    
    # 防止log(0)
    target_likelihood = max(target_likelihood, 1e-10)
    max_other = max(max_other, 1e-10)
    
    # Log-likelihood ratio（纯likelihood，不含prior）
    return np.log(target_likelihood) - np.log(max_other)


def optimize_legible_action(state, goals, target_idx, speed=100.0, n_samples=11):
    prior = np.ones(len(goals)) / len(goals)
    
    direction = goals[target_idx] - state
    dist = np.linalg.norm(direction)
    if dist < 1e-3:
        return np.array([0.0, 0.0])
    
    base_angle = np.arctan2(direction[1], direction[0])
    
    best_action = speed * direction / dist
    best_score = compute_legibility(best_action, state, goals, target_idx)
    
    for offset in np.linspace(-1.0, 1.0, n_samples):
        angle = base_angle + offset
        candidate = speed * np.array([np.cos(angle), np.sin(angle)])
        
        legibility = compute_legibility(candidate, state, goals, target_idx)
        task_score = -np.linalg.norm(goals[target_idx] - (state + candidate * 0.1))
        
        score = 500 * legibility + task_score
        
        if score > best_score:
            best_score = score
            best_action = candidate
    
    return best_action
