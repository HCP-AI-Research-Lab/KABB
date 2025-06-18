# my_moa_project/bayesian_mab.py

import random
import logging
import math
import time
from typing import List, Dict, Tuple

class KnowledgeAwareBayesianMAB:
    """
    Knowledge-aware Bayesian Multi-Armed Bandit (MAB) algorithm for dynamic expert selection.
    
    Before each sampling round, compute the knowledge distance based on:
      1. Concept overlap (expert.capabilities vs. knowledge_graph["concepts"])
      2. Path distance (expert["knowledge_dependencies"].length)
      3. Historical performance (alpha / (alpha + beta))
    Then combine to get distance, => knowledge_factor=exp(-distance)
    => Dynamically adjust alpha => Thompson sampling

    Also includes a "time decay" mechanism:
      - self.decay_factor**(time_elapsed) decays alpha, beta
    """

    def __init__(
        self,
        experts: List[dict],
        knowledge_graph: Dict,
        domain_label: str,
        difficulty: float,
        decay_factor: float = 0.99,
        weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
    ):
        """
        Initialize KnowledgeAwareBayesianMAB.

        :param experts: List of experts in the current domain.
        :param knowledge_graph: Knowledge graph of the domain.
        :param domain_label: Domain label.
        :param difficulty: Task difficulty.
        :param decay_factor: Time decay factor.
        :param weights: Weights for each factor in knowledge distance (overlap, path_dist, historical_sim).
        """
        self.experts = experts
        self.knowledge_graph = knowledge_graph
        self.domain_label = domain_label
        self.difficulty = difficulty
        self.decay_factor = decay_factor
        self.weights = weights

        self.expert_params = []  # Store (alpha, beta) for each expert
        self.last_update_time = []  # Record last update time for each expert

        # Expert ID mapping
        self.expert_id_map = {id(e): idx for idx, e in enumerate(self.experts)}

        # Initialization: read prior_alpha, prior_beta from config.yaml if available
        for e in self.experts:
            alpha = e.get("prior_alpha", 1.0)
            beta = e.get("prior_beta", 1.0)
            self.expert_params.append([alpha, beta])
            self.last_update_time.append(time.time())

    def compute_knowledge_distance(self, expert: dict, difficulty: float) -> float:
        """
        Compute the knowledge structure distance between the expert and the task.
        Larger distance => less suitable => lower factor.
        
        :param expert: Expert dictionary.
        :param difficulty: Task difficulty.
        :return: Distance value, larger means less matching.
        """
        # 1. Concept overlap
        caps = expert.get("capabilities", {})
        concepts = set(caps.keys())
        kg_concepts = set(self.knowledge_graph.get("concepts", []))

        if len(kg_concepts) > 0:
            overlap_ratio = len(concepts.intersection(kg_concepts)) / len(kg_concepts)
        else:
            overlap_ratio = 0.5  # If knowledge_graph has no concepts, default to 0.5

        # 2. Path distance
        dependencies = expert.get("knowledge_dependencies", [])
        if len(self.experts) > 0:
            path_dist = len(dependencies) / len(self.experts)
        else:
            path_dist = 1.0

        # 3. Historical performance similarity
        expert_idx = self.expert_id_map.get(id(expert), -1)
        if expert_idx == -1:
            logging.error("Expert index not found, possibly due to uninitialized expert.")
            historical_sim = 0.5
        else:
            alpha, beta = self.expert_params[expert_idx]
            historical_sim = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5

        # Combine all three => distance
        distance = (
            (1 - overlap_ratio) * self.weights[0] +
            path_dist * self.weights[1] +
            (1 - historical_sim) * self.weights[2]
        )

        # Adjust by task difficulty => log(1 + difficulty)
        distance *= math.log(1.0 + difficulty)

        logging.debug(f"Expert: {expert['name']}, Overlap Ratio: {overlap_ratio}, Path Dist: {path_dist}, "
                      f"Historical Sim: {historical_sim}, Distance: {distance}")

        return distance

    def sample_experts(self, top_k: int = 2) -> List[int]:
        """
        Select the top_k experts using time decay, knowledge distance, and Thompson sampling.

        :param top_k: Number of experts to select.
        :return: List of selected expert indices.
        """
        current_time = time.time()
        sample_scores = []

        for idx, exp in enumerate(self.experts):
            # Step1: Time decay
            time_elapsed = current_time - self.last_update_time[idx]
            alpha, beta = self.expert_params[idx]
            decay = self.decay_factor ** time_elapsed
            alpha *= decay
            beta *= decay
            self.expert_params[idx] = [alpha, beta]
            self.last_update_time[idx] = current_time

            logging.debug(f"Expert: {exp['name']}, Time Elapsed: {time_elapsed}, Decayed Alpha: {alpha}, Decayed Beta: {beta}")

            # Step2: Compute knowledge distance => factor
            dist = self.compute_knowledge_distance(exp, self.difficulty)
            knowledge_factor = math.exp(-dist)

            # Step3: Dynamic prior adjustment => adjusted_alpha = alpha * knowledge_factor
            adjusted_alpha = alpha * knowledge_factor

            # Step4: Thompson sampling
            sample_val = random.betavariate(adjusted_alpha, beta)
            sample_scores.append((sample_val, idx))

            logging.debug(f"Expert: {exp['name']}, Adjusted Alpha: {adjusted_alpha}, Beta: {beta}, Sample Val: {sample_val}")

        # Select Top_k
        sample_scores.sort(key=lambda x: x[0], reverse=True)
        selected = [s[1] for s in sample_scores[:top_k]]

        logging.info(f"[KnowledgeAwareBayesianMAB] sample_experts => selected indices: {selected}")
        return selected

    def update_expert(self, expert_idx: int, reward: float):
        """
        Update the alpha and beta parameters of the expert based on feedback.

        :param expert_idx: Expert index.
        :param reward: Reward value, recommended between 0 and 1.
        """
        if not (0.0 <= reward <= 1.0):
            logging.warning(f"Received reward {reward} out of bounds. Clamping to [0,1].")
            reward = max(0.0, min(1.0, reward))

        alpha, beta = self.expert_params[expert_idx]
        alpha += reward
        beta += (1.0 - reward)
        self.expert_params[expert_idx] = [alpha, beta]

        logging.info(f"[KnowledgeAwareBayesianMAB] update_expert => idx={expert_idx}, reward={reward}, new=(alpha={alpha}, beta={beta})")

    def get_params(self, expert_idx: int) -> Tuple[float, float]:
        """
        Get the (alpha, beta) parameters of the specified expert.

        :param expert_idx: Expert index.
        :return: (alpha, beta) tuple.
        """
        if 0 <= expert_idx < len(self.expert_params):
            return tuple(self.expert_params[expert_idx])
        else:
            logging.error(f"Invalid expert_idx={expert_idx}. Returning default (1.0, 1.0).")
            return (1.0, 1.0)