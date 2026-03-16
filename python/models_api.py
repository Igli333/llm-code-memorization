# import litellm thingy
import os
from dotenv import load_dotenv
import litellm
from litellm import completion

from tenacity import retry, stop_after_attempt, wait_exponential, after_log
import logging
logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True


# gpt-3.5 turbo
# gpt 4o-mini
# gpt 5-mini

# gemini 2.5 pro
# gemini 2.5 flash

load_dotenv()

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")

import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential
from func_timeout import func_timeout, FunctionTimedOut

class ModelApi():
    def __init__(self, model_name):
        self.model_name = model_name

    @retry(
        wait=wait_random_exponential(min=1, max=20), 
        stop=stop_after_attempt(3),
        reraise=True
    )
    def _execute_with_timeout(self, prompt):
        """
        Wraps the API call in a hard execution timeout.
        If completion() takes longer than 45s, func_timeout raises FunctionTimedOut.
        """
        return func_timeout(60, completion, kwargs={
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            # We set the internal timeout lower than our wrapper
            # "timeout": 40 
        })

    def infer(self, prompt):
        try:
            response = self._execute_with_timeout(prompt)
            return response.choices[0].message.content
        except FunctionTimedOut:
            return "Error: The operation timed out globally."
        except Exception as e:
            return f"Error: Failed after retries. {str(e)}"