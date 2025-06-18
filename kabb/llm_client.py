import os
import asyncio
import logging
from typing import List, Union, Optional, Any
import together
from together import AsyncTogether

logging.basicConfig(level=logging.INFO)


class LLMClient:
    """
    Wrapper for interacting with the Together API.
    """

    def __init__(
        self,
        api_key: Union[str, List[str]],
        provider: str = "together",
        max_retries: int = 4,
        retry_backoff: List[int] = [1, 2, 4, 8],
        timeout: int = 60,  # Request timeout in seconds
    ):
        """
        Initialize LLMClient.

        :param api_key: API key, can be a single string or a list of strings.
        :param provider: LLM provider name.
        :param max_retries: Maximum number of retries.
        :param retry_backoff: List of wait times for retries, in seconds.
        :param timeout: Request timeout in seconds.
        """
        if api_key is None or api_key == "":
            api_key = os.environ.get("TOGETHER_API_KEY")
        if isinstance(api_key, str):
            self.api_key = [api_key]  # Convert single string to a single-element list
        elif isinstance(api_key, list) and all(isinstance(key, str) for key in api_key):
            self.api_key = api_key
        else:
            raise ValueError("api_key must be a string or a list of strings.")

        self.current_key_index = 0
        self.provider = provider.lower()
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout = timeout

        if self.provider == "together":
            self.client = AsyncTogether(api_key=self.api_key[self.current_key_index])
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def switch_api_key(self) -> bool:
        """
        Switch to the next API key. Returns False if no more keys are available.

        :return: Whether successfully switched to the next API key.
        """
        if self.current_key_index < len(self.api_key) - 1:
            self.current_key_index += 1
            new_api_key = self.api_key[self.current_key_index]
            self.client = AsyncTogether(api_key=new_api_key)
            logging.warning(f"Switched to next API key: {new_api_key}")
            return True
        else:
            logging.error("All API keys have been exhausted.")
            return False

    async def run_llm(
        self,
        model: str,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        stream: bool = True,
    ) -> Optional[Any]:
        """
        Call Together API's chat.completions.create method, with retry and timeout mechanism.

        :param model: Model name to use.
        :param messages: List of messages, format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}].
        :param temperature: Temperature parameter, controls randomness of generation.
        :param max_tokens: Maximum number of tokens to generate.
        :param stream: Whether to enable streaming response.
        :return: API response object or None (if failed).
        """
        while self.current_key_index < len(self.api_key):
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=stream,
                        ),
                        timeout=self.timeout,
                    )
                    logging.info(f"LLM API call successful on attempt {attempt}.")
                    return response
                except together.error.APIError as e:
                    if "Credit limit exceeded" in e._message or e._type == "credit_limit":
                        logging.error(f"API key credit limit exceeded: {e}.")
                        if not self.switch_api_key():
                            return None
                        break  # Exit current attempt loop, switch to new key and retry
                    elif attempt < self.max_retries:
                        sleep_time = self.retry_backoff[min(attempt - 1, len(self.retry_backoff) - 1)]
                        logging.warning(f"LLM API error: {e}. Retrying in {sleep_time}s (Attempt {attempt}/{self.max_retries})...")
                        await asyncio.sleep(sleep_time)
                    else:
                        logging.error(f"LLM API error after {self.max_retries} attempts: {e}.")
                except together.error.RateLimitError as e:
                    if attempt < self.max_retries:
                        sleep_time = self.retry_backoff[min(attempt - 1, len(self.retry_backoff) - 1)]
                        logging.warning(f"Rate limit exceeded: {e}. Retrying in {sleep_time}s (Attempt {attempt}/{self.max_retries})...")
                        await asyncio.sleep(sleep_time)
                    else:
                        logging.error(f"Rate limit exceeded after {self.max_retries} attempts: {e}.")
                except asyncio.TimeoutError:
                    if attempt < self.max_retries:
                        sleep_time = self.retry_backoff[min(attempt - 1, len(self.retry_backoff) - 1)]
                        logging.warning(f"LLM API call timed out. Retrying in {sleep_time}s (Attempt {attempt}/{self.max_retries})...")
                        await asyncio.sleep(sleep_time)
                    else:
                        logging.error(f"LLM API call timed out after {self.max_retries} attempts.")
                except Exception as e:
                    if attempt < self.max_retries:
                        sleep_time = self.retry_backoff[min(attempt - 1, len(self.retry_backoff) - 1)]
                        logging.error(f"Unexpected error: {e}. Retrying in {sleep_time}s (Attempt {attempt}/{self.max_retries})...")
                        await asyncio.sleep(sleep_time)
                    else:
                        logging.error(f"Unexpected error after {self.max_retries} attempts: {e}.")
            else:
                # All retries for current API key exhausted, switch API key
                if not self.switch_api_key():
                    return None
        logging.error("All API keys have been exhausted.")
        return None

    # async def _process_chunk(self, chunk: Any) -> Optional[str]:
    #     """
    #     Process a single data chunk.
    #     """
    #     try:
    #         if not hasattr(chunk, 'choices') or not chunk.choices:
    #             return None
    #         
    #         choice = chunk.choices[0]
    #         
    #         # Only check finish_reason when delta.content is empty
    #         if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
    #             if choice.delta.content:
    #                 return choice.delta.content
    #             # When content is empty, possibly the last chunk, check finish_reason
    #             elif hasattr(choice, 'finish_reason'):
    #                 if choice.finish_reason == '':
    #                     choice.finish_reason = 'stop'
    #                 elif choice.finish_reason not in ['length', 'stop', 'eos', 'tool_calls', 'error']:
    #                     logging.warning(f"Invalid finish_reason: {choice.finish_reason}, setting to 'stop'")
    #                     choice.finish_reason = 'stop'
    #         return None
    #         
    #     except Exception as e:
    #         logging.error(f"Error processing chunk content: {e}")
    #         raise