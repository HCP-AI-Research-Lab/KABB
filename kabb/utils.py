# my_moa_project/utils.py

import random
import torch
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np
import yaml
import logging
import re
import os
from kabb.bandit import KnowledgeAwareBayesianMAB
# Global SentenceTransformer instance to avoid repeated loading
EMBEDDER = SentenceTransformer(
    os.getenv('SENTENCE_TRANSFORMER_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
)



def logic_based_domain_inference(user_input: str, domain_config: dict, top_k: int = 3) -> Tuple[List[str], str]:
    """
    Improved version: returns the names of the top K most relevant domains, without returning scores.
    Sorts domains by score in descending order and returns the names of the top K domains.
    
    :param user_input: User input text
    :param domain_config: Domain inference configuration dictionary
    :param top_k: Number of most relevant domains to return
    :return: (top_k_domains, cleaned_input) tuple
    """
    # Convert input to lowercase for keyword matching
    input_lower = user_input.lower()
    # Store original input
    original_input = user_input

    # 1) Compute sentence embedding for similarity comparison with each domain's typical_samples
    user_emb = EMBEDDER.encode([user_input], convert_to_tensor=True)[0]

    # 2) Iterate over each domain, compute base score + keyword bonus
    scores = {}
    cleaned_inputs = {}

    for domain, settings in domain_config.items():
        # 2.1) Prior score
        prior = settings.get("prior", 0.3)

        # 2.2) Sentence embedding similarity with typical_samples
        sample_text = settings.get("typical_samples", "")
        sample_emb = EMBEDDER.encode([sample_text], convert_to_tensor=True)[0]
        sim = float(torch.nn.functional.cosine_similarity(user_emb, sample_emb, dim=0))

        # Base score = prior + sim
        score = prior + sim * 1.2

        # 2.3) Keyword bonus (not limited to "prefix", but checks if it appears in input)
        symbol_patterns = settings.get("symbol_patterns", [])
        keyword_bonus = 0.0

        for pat in symbol_patterns:
            # For flexibility, use \b boundary match, or just "if pat.lower() in input_lower"
            pattern = r"\b" + re.escape(pat.lower()) + r"\b"
            matches = re.findall(pattern, input_lower)
            if matches:
                # Add 0.6 for each match
                keyword_bonus += 0.6 * len(matches)

        score += keyword_bonus

        # 2.4) Record the score for each domain
        scores[domain] = score

        # 2.5) Keep the cleaned_input logic here
        cleaned_inputs[domain] = original_input

    # 3) Sort by score and get the top K domains
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    logging.info(f"[logic_based_domain_inference] => domain_scores={scores}, top_k_domains={sorted_scores}")

    # 4) Get the names of the top K domains
    top_k_domains = [domain for domain, _ in sorted_scores]

    # 5) Return (top K domains, corresponding processed input)
    return top_k_domains, cleaned_inputs[top_k_domains[0]]

def estimate_task_difficulty(user_input: str, domain_setting: dict) -> float:
    """
    Estimate task difficulty based on text length, and clamp to domain_setting['difficulty_range'].

    :param user_input: User input text.
    :param domain_setting: Current domain's configuration dictionary.
    :return: float difficulty value.
    """
    difficulty_range = domain_setting.get("difficulty_range", [1, 5])
    min_diff, max_diff = difficulty_range[0], difficulty_range[1]
    input_len = len(user_input)

    # Use (input_len / 100) => 1 ~ 10
    base_diff = min(max(input_len / 100, 1.0), 10.0)
    # Map to [min_diff, max_diff]
    diff_value = (base_diff / 10.0) * (max_diff - min_diff) + min_diff
    logging.info(f"[estimate_task_difficulty] input_len={input_len}, base_diff={base_diff}, final_diff={diff_value}")
    return diff_value

##################################################
# Multi-Agent Reinforcement Learning (MARL) Example
##################################################

class MultiAgentCoordinator:
    """
    Multi-agent coordinator example: each expert corresponds to an Agent, or a central Agent manages experts.
    Currently just an example, implements simple expert selection logic.
    """

    def __init__(self, experts: List[dict], knowledge_graph: dict, domain_label: str, difficulty: float):
        """
        Initialize MultiAgentCoordinator.

        :param experts: List of experts in the current domain.
        :param knowledge_graph: Knowledge graph of the domain.
        :param domain_label: Domain label.
        :param difficulty: Task difficulty.
        """
        self.experts = experts
        self.knowledge_graph = knowledge_graph
        self.domain_label = domain_label
        self.difficulty = difficulty
        # TODO: RL policy/env can be initialized here

    def select_experts(self, user_input: str) -> List[int]:
        """
        Logic for selecting experts. Currently just an example, defaults to selecting the first 2 experts.

        :param user_input: User input.
        :return: List of selected expert indices.
        """
        logging.info("[MultiAgentCoordinator] (MARL) => Example: Default select first 2 experts")
        return [0, 1] if len(self.experts) >= 2 else list(range(len(self.experts)))

##################################################
# Expert result integration, knowledge conflict detection, etc.
##################################################

def compute_knowledge_transfer(partial_responses: List[str], experts_pool: List[dict]) -> str:
    """
    Concatenate all experts' partial responses as collective knowledge output.
    If more complex "order" or "multi-round collaboration" is needed, handle it here.

    :param partial_responses: List of expert responses.
    :param experts_pool: List of experts in the current domain.
    :return: Integrated content.
    """
    integrated = "\n\n".join(partial_responses)
    return integrated

def evaluate_complementarity(selected_experts: List[dict], knowledge_graph: dict) -> float:
    """
    Calculate the complementarity score of selected experts.

    Count all capability keys of experts, less overlap => higher complementarity.
    Result range [0,1]

    :param selected_experts: List of selected experts.
    :param knowledge_graph: Knowledge graph of the domain.
    :return: Complementarity score.
    """
    if not selected_experts:
        return 1.0

    capability_keys = set()
    for e in selected_experts:
        caps = e.get("capabilities", {})
        for k in caps.keys():
            capability_keys.add(k)

    # Simple: use total number of capabilities of selected_experts / number of selected_experts => roughly measures redundancy
    total_cap = sum(len(e.get("capabilities", {})) for e in selected_experts)

    if len(capability_keys) == 0:
        return 1.0

    overlap = total_cap - len(capability_keys)
    max_overlap = len(selected_experts) * (len(selected_experts) - 1) / 2.0
    if max_overlap == 0:
        return 1.0
    complementarity = 1.0 - (overlap / max_overlap)
    return max(0.0, min(1.0, complementarity))

def evaluate_complementarity(selected_experts: List[dict], knowledge_graph: dict) -> float:
    """
    Evaluate the complementarity of the selected expert combination.

    The complementarity score is based on the experts' capability vectors and concept relationships in the knowledge graph.
    Returns a score between 0 and 1, where 1 means highly complementary.

    :param selected_experts: List of selected experts.
    :param knowledge_graph: Knowledge graph of the domain.
    :return: Complementarity score.
    """
    capability_keys = set()
    for expert in selected_experts:
        capability_keys.update(expert.get("capabilities", {}).keys())

    capability_keys = list(capability_keys)

    # Build capability matrix
    capability_matrix = []
    for expert in selected_experts:
        capabilities = expert.get("capabilities", {})
        capability_vector = [capabilities.get(k, 0) for k in capability_keys]
        capability_matrix.append(capability_vector)
    capability_matrix = np.array(capability_matrix)

    # Calculate capability overlap
    overlap = np.dot(capability_matrix, capability_matrix.T)
    np.fill_diagonal(overlap, 0)  # Exclude self-overlap
    total_overlap = np.sum(overlap)

    # Calculate complementarity score
    max_overlap = len(selected_experts) * (len(selected_experts) - 1) * 1.0  # Maximum overlap per pair is 1
    complementarity_score = 1 - (total_overlap / max_overlap) if max_overlap > 0 else 1.0
    return complementarity_score

def detect_knowledge_conflict(selected_experts: List[dict], knowledge_graph: dict) -> bool:
    """
    Detect knowledge conflicts in the selected expert combination.

    :param selected_experts: List of selected experts.
    :param knowledge_graph: Knowledge graph of the domain.
    :return: Whether there is a conflict.
    """
    conflicts = knowledge_graph.get("conflicts", [])
    expert_names = [e["name"] for e in selected_experts]

    for pair in conflicts:
        if len(pair) >= 2:
            if pair[0] in expert_names and pair[1] in expert_names:
                logging.warning(f"Detect conflict between {pair[0]} and {pair[1]}")
                return True

    return False

class KnowledgeManager:
    """
    Responsible for dynamically updating domain knowledge graphs, expert capability models, and knowledge dependencies.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize KnowledgeManager.

        :param config_path: Path to configuration file.
        """
        self.config_path = config_path
        self.load_knowledge_graph()

    def load_knowledge_graph(self):
        """
        Load knowledge graph configuration.
        """
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.knowledge_graph = config.get("knowledge_graph", {})

    def update_knowledge_graph(self, domain: str, new_concept: str, new_task_pattern: str):
        """
        Dynamically update the domain knowledge graph.

        :param domain: Domain label.
        :param new_concept: New concept to add.
        :param new_task_pattern: New task pattern to add.
        """
        if domain not in self.knowledge_graph:
            self.knowledge_graph[domain] = {"concepts": [], "task_patterns": [], "conflicts": []}
        if new_concept and new_concept not in self.knowledge_graph[domain].get("concepts", []):
            self.knowledge_graph[domain]["concepts"].append(new_concept)
        if new_task_pattern and new_task_pattern not in self.knowledge_graph[domain].get("task_patterns", []):
            self.knowledge_graph[domain]["task_patterns"].append(new_task_pattern)
        self.save_config()

    def update_expert_capabilities(self, expert_name: str, new_capabilities: Dict[str, float]):
        """
        Dynamically update the expert's capability vector.

        :param expert_name: Expert name.
        :param new_capabilities: Dictionary of new or updated capabilities.
        """
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        updated = False
        for domain, experts in config.get("experts", {}).items():
            for expert in experts:
                if expert["name"] == expert_name:
                    expert["capabilities"].update(new_capabilities)
                    updated = True
                    break
            if updated:
                break
        if updated:
            self.save_config()
        else:
            logging.warning(f"Expert '{expert_name}' not found in config.")

    def save_config(self):
        """
        Save the updated configuration file.
        """
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump({"knowledge_graph": self.knowledge_graph}, f)
        self.load_knowledge_graph()

def simple_domain_classifier(user_input: str, domain_config: dict) -> str:
    """
    Use domain keywords for the simplest keyword matching to determine the task domain.

    :param user_input: User input text.
    :param domain_config: Domain inference configuration dictionary.
    :return: Determined domain label.
    """
    words = user_input.split()
    best_domain = "academic_paper"
    max_hits = 0
    for domain_label, settings in domain_config.items():
        keywords = settings.get("symbol_patterns", [])
        hits = sum(1 for w in words if any(k.lower() == w.lower() for k in keywords))
        if hits > max_hits:
            max_hits = hits
            best_domain = domain_label
    return best_domain

def bayesian_domain_inference(task_features: List[str], domain_prior: dict, domain_keywords: dict) -> dict:
    """
    Update the probability that a task belongs to each domain using Bayesian updating.

    :param task_features: List of task keywords.
    :param domain_prior: Prior probability for each domain.
    :param domain_keywords: List of keywords for each domain.
    :return: Posterior probability for each domain.
    """
    posterior = {}
    for domain, prior in domain_prior.items():
        keywords = domain_keywords.get(domain, [])
        likelihood = sum(1 for feature in task_features if any(k in feature for k in keywords))
        # Use Laplace smoothing
        posterior[domain] = prior * (likelihood + 1)
    # Normalize
    total = sum(posterior.values())
    for domain in posterior:
        posterior[domain] /= total
    return posterior

def select_domain(posterior: dict) -> str:
    """
    Select the most likely domain based on posterior probability.

    :param posterior: Posterior probability dictionary.
    :return: Selected domain label.
    """
    return max(posterior, key=posterior.get)

def combine_moa_bayesian(moa_weights: List[float], mab: KnowledgeAwareBayesianMAB, top_k: int = 2) -> List[int]:
    """
    Combine MoA weights with Bayesian MAB sampled values to select experts.

    :param moa_weights: List of MoA weights.
    :param mab: KnowledgeAwareBayesianMAB instance.
    :param top_k: Number of experts to select.
    :return: List of selected expert indices.
    """
    samples = []
    for i, w in enumerate(moa_weights):
        alpha, beta = mab.get_params(i)
        sample_val = random.betavariate(alpha, beta)  # Thompson sampling
        combined_score = sample_val * w
        samples.append((combined_score, i))
    # Sort by combined_score descending, select top_k
    samples.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in samples[:top_k]]

##################################################
# Mixture of Agents (MoA) Layer
##################################################

class MoALayer(torch.nn.Module):
    """
    Simple Mixture of Agents (MoA) layer:
    Maps input embedding linearly to num_agents dimensions, then applies softmax.
    """

    def __init__(self, d_model: int, num_agents: int):
        """
        Initialize MoALayer.

        :param d_model: Dimension of input embedding.
        :param num_agents: Number of agents.
        """
        super(MoALayer, self).__init__()
        self.num_agents = num_agents
        self.gate = torch.nn.Linear(d_model, num_agents)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: Input tensor, shape (batch_size, d_model).
        :return: Agent weights, shape (batch_size, num_agents).
        """
        gate_values = self.gate(x)
        agent_weights = self.softmax(gate_values)
        return agent_weights