import numpy as np
from .legibility import pi_H, compute_legibility, compute_reward
from config import (CONTROL_SPEED, BETA_BASE, BETA_MIN, B_THRESH, 
                   D_MIN, D_MAX, TASK_WEIGHT, SEARCH_ANGLES, ANGLE_RANGE, 
                   BETA_RATIONALITY, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX)


class LegibleSharedAutonomy:
    def __init__(self, goals, task_weight=None):
        self.goals = goals
        self.beliefs = np.ones(len(goals)) / len(goals)
        self.beta = BETA_BASE
        self.d_workspace = np.sqrt((X_MAX - X_MIN)**2 + (Y_MAX - Y_MIN)**2 + (Z_MAX - Z_MIN)**2)
        self.task_weight = task_weight if task_weight is not None else TASK_WEIGHT
        self.started = False
    
    def update_belief(self, state, user_input):
        if np.linalg.norm(user_input) < 0.01:
            return
        
        self.started = True
        
        log_likelihoods = []
        for g in self.goals:
            user_reward = compute_reward(user_input, state, g)
            log_likelihoods.append(BETA_RATIONALITY * user_reward)
        
        log_likelihoods = np.array(log_likelihoods)
        
        log_beliefs = np.log(self.beliefs + 1e-100)
        log_beliefs += log_likelihoods
        
        max_log = np.max(log_beliefs)
        log_beliefs -= max_log
        
        self.beliefs = np.exp(log_beliefs)
        self.beliefs /= np.sum(self.beliefs)
    
    def compute_adaptive_beta(self, state, target_goal):
        b_max = np.max(self.beliefs)
        dist = np.linalg.norm(target_goal - state)
        
        alpha = np.clip((b_max - B_THRESH) / (1.0 - B_THRESH), 0.0, 1.0)
        
        d_min_abs, d_max_abs = D_MIN * self.d_workspace, D_MAX * self.d_workspace
        gamma = np.clip((d_max_abs - dist) / (d_max_abs - d_min_abs), 0.0, 1.0)
        
        return BETA_BASE - alpha * gamma * (BETA_BASE - BETA_MIN)
    
    def compute_robot_action(self, state, user_input):
        target_idx = np.argmax(self.beliefs)
        target_goal = self.goals[target_idx]
        
        direction = target_goal - state
        dist = np.linalg.norm(direction)
        if dist < 1e-3:
            return np.zeros(3)
        
        task_action = CONTROL_SPEED * direction / dist
        self.beta = self.compute_adaptive_beta(state, target_goal)
        
        best_action, best_score = task_action, -np.inf
        
        task_dir = direction / dist
        
        for i in range(SEARCH_ANGLES):
            theta = (i / (SEARCH_ANGLES - 1) - 0.5) * 2 * ANGLE_RANGE
            phi = (i / (SEARCH_ANGLES - 1) - 0.5) * 2 * ANGLE_RANGE
            
            perp1 = np.array([-task_dir[1], task_dir[0], 0])
            if np.linalg.norm(perp1) < 1e-6:
                perp1 = np.array([1, 0, 0])
            perp1 = perp1 / np.linalg.norm(perp1)
            
            perp2 = np.cross(task_dir, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)
            
            rot_vec = task_dir + theta * perp1 + phi * perp2
            rot_vec = rot_vec / np.linalg.norm(rot_vec)
            
            candidate = CONTROL_SPEED * rot_vec
            
            leg_score = compute_legibility(candidate, state, self.goals, target_idx)
            task_score = -np.linalg.norm(target_goal - (state + candidate * 0.1))
            score = self.task_weight * leg_score + task_score
            
            if score > best_score:
                best_score, best_action = score, candidate
        
        return best_action
    
    def reset(self):
        self.beliefs = np.ones(len(self.goals)) / len(self.goals)
        self.beta = BETA_BASE
        self.started = False