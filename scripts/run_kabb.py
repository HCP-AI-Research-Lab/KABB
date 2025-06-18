import sys
import os

# Add project root to sys.path for kabb package import
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import List, Dict
import yaml
import asyncio
import logging
import argparse
from typing import Tuple, List
import json
import time
import fcntl

# Set HuggingFace endpoint mirror if needed
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from sentence_transformers import SentenceTransformer

from kabb.bandit import KnowledgeAwareBayesianMAB
from kabb.llm_client import LLMClient
from kabb.utils import (
    logic_based_domain_inference,
    compute_knowledge_transfer,
    evaluate_complementarity,
    detect_knowledge_conflict,
    estimate_task_difficulty
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("moa_system.log"),
        logging.StreamHandler()
    ]
)

USE_MARL = True

DEFAULT_DOMAIN = "problem_solving_and_decision_making"
DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "config_math_template.yaml")

GENERATOR = "kabb_default"
OUTPUT_DIR = ""
OUTPUT_FILE = "kabb_outputs.json"
START_IND = 0
END_IND = 805
SLEEP_TIME = 0.1

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run Multi-Agent Collaboration System (MoA)")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to config file"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="",
        help="Question or task description to process"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory"
    )
    parser.add_argument(
        "--dataset_dirs",
        type=str,
        nargs='+',
        default=[],
        help="List of dataset directories"
    )
    parser.add_argument(
        "--domain_dirs",
        type=str,
        nargs='+',
        default=[],
        help="List of domain directories"
    )
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """
    Load configuration file.

    :param config_path: Path to config file.
    :return: Config dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

async def generate_response(llm_client: LLMClient, model: str, messages: List[dict], role_name: str, max_tokens: int = 4000) -> str:
    """
    Interact with LLM to generate text.

    :param llm_client: LLMClient instance.
    :param model: Model name.
    :param messages: List of messages.
    :param role_name: Role name for logging.
    :param max_tokens: Maximum number of tokens to generate.
    :return: Generated text.
    """
    logging.info(f"[{role_name}] Generating response...")
    response_stream = await llm_client.run_llm(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )
    if not response_stream:
        logging.error(f"[{role_name}] Response generation failed.")
        return ""
    
    role_response = ""
    try:
        async for chunk in response_stream:
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
            role_response += content
    except Exception as e:
        logging.error(f"[{role_name}] Error during generation: {e}")
    print("\n")
    return role_response

async def get_expert_responses(selected_indices: List[int], domain_experts: List[dict], system_prompt: str, llm_client: LLMClient, user_input: str) -> List[str]:
    """
    Generate expert responses in parallel.

    :param selected_indices: List of selected expert indices.
    :param domain_experts: List of experts for the current domain.
    :param system_prompt: System prompt.
    :param llm_client: LLMClient instance.
    :param user_input: User input.
    :return: List of expert responses.
    """
    tasks = []
    for idx in selected_indices:
        if idx >= len(domain_experts) or idx < 0:
            logging.error(f"Selected index {idx} out of range, skipping expert.")
            continue
        expert = domain_experts[idx]
        role_name = expert["name"]
        role_prompt = expert["role_prompt"]
        model_name = expert.get("model", "Qwen/Qwen2-72B-Instruct")
        task_template = expert["task_template"]
        user_content = task_template.replace("{{user_prompt}}", user_input)

        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n{role_prompt}"},
            {"role": "user", "content": user_content}
        ]
        tasks.append(generate_response(
            llm_client, model_name, messages, role_name, max_tokens=expert.get("max_tokens", 8000)
        ))
    partial_responses = await asyncio.gather(*tasks)
    return partial_responses

async def get_domain_all_expert_responses(
    domain_label: str,
    domain_experts_for_label: List[dict],
    system_prompt: str,
    llm_client: LLMClient,
    user_input: str
) -> List[str]:
    """
    Generate responses from all experts in a domain in parallel, preserving expert order.

    :param domain_label: Domain name.
    :param domain_experts_for_label: List of expert info for the domain.
    :param system_prompt: System prompt.
    :param llm_client: LLMClient instance.
    :param user_input: User input.
    :return: List of responses from all experts in the domain, in order.
    """
    async def get_expert_response(index: int, expert: dict) -> tuple:
        """
        Get a single expert's response and return its index and response.

        :param index: Index of the expert.
        :param expert: Expert info.
        :return: (index, response)
        """
        role_name = expert["name"]
        role_prompt = expert["role_prompt"]
        model_name = expert.get("model", "Qwen/Qwen2-72B-Instruct")
        task_template = expert["task_template"]
        user_content = task_template.replace("{{user_prompt}}", user_input)

        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n{role_prompt}"},
            {"role": "user", "content": user_content},
        ]

        response = await generate_response(
            llm_client, model_name, messages, role_name, max_tokens=expert.get("max_tokens", 8000)
        )

        logging.info(f"Expert={role_name} Messages={messages} Response={response}")

        return index, response

    tasks = [
        get_expert_response(index, expert)
        for index, expert in enumerate(domain_experts_for_label)
    ]

    results = await asyncio.gather(*tasks)

    sorted_responses = [response for index, response in sorted(results, key=lambda x: x[0])]

    return sorted_responses

def get_integrated_content(partial_responses: List[str], domain_experts: List[dict]) -> str:
    """
    Process and integrate expert responses.

    :param partial_responses: List of expert responses.
    :param domain_experts: List of experts for the current domain.
    :return: Integrated content.
    """
    integrated_content = compute_knowledge_transfer(partial_responses, domain_experts)
    return integrated_content

def get_integrated_content_new(all_responses: Dict[str, List[str]], selected_experts: List[dict], use_contextual_expert: bool = False) -> str:
    """
    Integrate all responses from all domains and experts, return the final integrated content.

    :param all_responses: Dict of domain to list of expert responses.
    :param selected_experts: List of all selected experts.
    :param use_contextual_expert: Whether to use contextual expert for further integration.
    :return: Integrated content string.
    """
    integrated_content = []
    for domain_label, responses in all_responses.items():
        for i, response in enumerate(responses):
            expert = selected_experts[i]
            expert_name = expert["name"]
            integrated_content.append(f"domain: {domain_label} | expert: {expert_name} \nresponse: {response}\n")
    if use_contextual_expert:
        integrated_content_summary = get_contextual_summary(integrated_content)
        return integrated_content_summary
    return "\n".join(integrated_content)

def get_contextual_summary(integrated_content: List[str]) -> str:
    """
    Example implementation of a contextual expert for integrating all responses.

    :param integrated_content: List of all expert responses.
    :return: Contextually integrated content.
    """
    return "\n".join(integrated_content)

async def integrate_expert_responses(integrator_info: dict, system_prompt: str, llm_client: LLMClient, user_input: str, integrated_content: str) -> str:
    """
    Call the integrator to generate the final answer.

    :param integrator_info: Integrator config info.
    :param system_prompt: System prompt.
    :param llm_client: LLMClient instance.
    :param user_input: User input.
    :param integrated_content: Integrated expert content.
    :return: Final answer.
    """
    integrator_role_name = integrator_info["name"]
    integrator_role_prompt = integrator_info["role_prompt"]
    integrator_model = integrator_info.get("model", "Qwen/Qwen2-72B-Instruct")
    integrator_task_template = integrator_info["task_template"]
    final_user_content = integrator_task_template.replace("{{user_prompt}}", user_input).replace("{{expert_proposals}}", integrated_content)

    integrator_messages = [
        {"role": "system", "content": f"{system_prompt}\n\n{integrator_role_prompt}"},
        {"role": "user", "content": final_user_content}
    ]
    final_answer = await generate_response(
        llm_client,
        integrator_model,
        integrator_messages,
        integrator_role_name,
        max_tokens=integrator_info.get("max_tokens", 8000)
    )

    logging.info(f"Integrator={integrator_role_name} Messages={integrator_messages} Response={final_answer}")

    return final_answer

def get_user_feedback(selected_indices: List[int], domain_experts: List[dict], mab: KnowledgeAwareBayesianMAB):
    """
    Get user feedback for experts and update the MAB model.

    :param selected_indices: List of selected expert indices.
    :param domain_experts: List of experts for the current domain.
    :param mab: KnowledgeAwareBayesianMAB instance.
    """
    for idx in selected_indices:
        expert = domain_experts[idx]
        score_str = input(f"Score for expert [{expert['name']}] (0~1, 1=excellent): ")
        try:
            base_reward = float(score_str)
            base_reward = max(0.0, min(1.0, base_reward))
        except ValueError:
            base_reward = 0.5
        mab.update_expert(idx, base_reward)

def get_domain_stats(domain_path: str) -> Tuple[int, int, int]:
    """
    Get processing statistics for a given domain directory.
    
    :param domain_path: Path to domain directory.
    :return: (total files, processed files, remaining files)
    """
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                output = json.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    else:
        output = {}

    domain_name = os.path.basename(domain_path)
    processed_files = {
        file_key.split('_')[2] 
        for file_key in output.keys() 
        if domain_name in file_key
    }
    all_files = set(f for f in os.listdir(domain_path) if f.endswith('.json'))
    total = len(all_files)
    processed = len(processed_files)
    remaining = total - processed
    return total, processed, remaining

def process_domain(domain_path: str):
    """
    Process all files in a single domain.

    :param domain_path: Path to domain directory.
    """
    domain_name = os.path.basename(domain_path)
    logging.info(f"Start processing domain: {domain_name}")
    asyncio.run(main(domain_path=domain_path))

async def main():
    """
    Main function to execute the multi-agent collaboration workflow.
    """
    args = parse_arguments()
    config = load_config(args.config)

    if args.question:
        user_input = args.question
    else:
        user_input = input("Input your question: ")
    
    domain_settings = config.get("domain_inference_settings", {})
    knowledge_graph = config.get("knowledge_graph", {})
    experts_config = config.get("experts", {})
    integrator_config = config.get("integrator", {})
    system_prompt = config["system_prompts"]["default"]

    domain_label, cleaned_input = logic_based_domain_inference(user_input, domain_settings, top_k=2)
    logging.info(f"[Domain Inference] logic-based selected domain: {domain_label}")

    if isinstance(domain_label, list):
        missing_labels = [label for label in domain_label if label not in experts_config]
        if missing_labels:
            logging.warning(f"domain_label(s) not in experts_config: {missing_labels}, fallback => DEFAULT_DOMAIN")
            domain_label = DEFAULT_DOMAIN
    else:
        if domain_label not in experts_config:
            logging.warning(f"domain_label={domain_label} not in experts_config, fallback => DEFAULT_DOMAIN")
            domain_label = DEFAULT_DOMAIN

    if isinstance(domain_label, list):
        difficulty = estimate_task_difficulty(cleaned_input, domain_settings.get(domain_label[0], {}))
    else:
        difficulty = estimate_task_difficulty(cleaned_input, domain_settings.get(domain_label, {}))
    logging.info(f"[Task Difficulty] difficulty={difficulty}")

    if isinstance(domain_label, list):
        domain_experts = []
        for label in domain_label:
            experts = experts_config.get(label, [])
            if experts:
                domain_experts.append((label, experts))
            else:
                logging.warning(f"Domain {label} not found in experts_config")
    else:
        domain_experts = experts_config.get(domain_label, [])
        if not domain_experts:
            logging.warning(f"Domain {domain_label} not found in experts_config")

    logging.info(f"Loaded expert info: {domain_experts}")

    integrator_info = integrator_config.get("default")

    if not domain_experts or not integrator_info:
        logging.error(f"Expert or integrator not defined for domain={domain_label}, fallback => DEFAULT_DOMAIN")
        domain_label = DEFAULT_DOMAIN
        domain_experts = experts_config.get(DEFAULT_DOMAIN, [])
        integrator_info = integrator_config.get(DEFAULT_DOMAIN, integrator_config.get("default"))

    try:
        api_key = os.environ.get("TOGETHER_API_KEY") or config.get("llm", {}).get("api_key")
        if not api_key:
            logging.error("API key not found. Please set TOGETHER_API_KEY environment variable or specify in config file.")
            return
        llm_client = LLMClient(api_key=api_key)
    except ValueError as e:
        logging.error(e)
        return

    selected_indices = []

    if isinstance(domain_experts, list) and len(domain_experts) > 0 and isinstance(domain_experts[0], tuple):
        domain_experts_for_selection = dict(domain_experts)
    else:
        domain_experts_for_selection = {domain_label: domain_experts}

    if isinstance(domain_label, list):
        selected_experts = []
        for label in domain_label:
            experts_for_label = domain_experts_for_selection.get(label, [])
            selected_experts.extend(experts_for_label)
    else:
        selected_experts = domain_experts_for_selection.get(domain_label, [])

    logging.info(f"Selected experts: {[expert.get('name', 'Unknown') for expert in selected_experts]}")

    logging.info(f"User Input: {cleaned_input}")

    if isinstance(domain_label, list):
        all_responses = {}
        for label in domain_label:
            domain_experts_for_label = [expert for expert in selected_experts 
                                      if label in expert.get('domain_expertise', [])]
            if domain_experts_for_label:
                domain_responses = await get_domain_all_expert_responses(
                    label, domain_experts_for_label, system_prompt, llm_client, cleaned_input
                )
                all_responses[label] = domain_responses
    else:
        domain_responses = await get_domain_all_expert_responses(
            domain_label, selected_experts, system_prompt, llm_client, cleaned_input
        )
        all_responses = {domain_label: domain_responses}

    logging.info(f"All Expert Responses: {all_responses}")
    
    integrated_content = get_integrated_content_new(all_responses, selected_experts, use_contextual_expert=False)

    logging.info(f"All Expert Responses after integrated: {integrated_content}")

    final_answer = await integrate_expert_responses(integrator_info, system_prompt, llm_client, cleaned_input, integrated_content)

    print("\n==== Final Output ====\n")
    print(final_answer)

    print("\n==== Processing Complete ====\n")

if __name__ == "__main__":
    asyncio.run(main())