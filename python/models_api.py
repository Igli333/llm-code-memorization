# import litellm thingy
import os
from dotenv import load_dotenv
import litellm
from litellm import completion

litellm.suppress_debug_info = True


# gpt-3.5 turbo
# gpt 4o-mini
# gpt 5-mini

# gemini 2.5 pro
# gemini 2.5 flash

load_dotenv()

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")


class ModelApi():
    def __init__(self, model_name):
        self.model_name = model_name

    def infer(self, prompt):
        response = completion(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            num_retries=5,
            timeout=60,
        )

        return response.choices[0].message.content
