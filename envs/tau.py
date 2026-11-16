from tau_bench.envs import get_env
from tau_bench.envs.user import LLMUserSimulationEnv


def make_env(domain, split, task_id):
    # Explicit task_index avoids upstream randint(0, len(tasks)) inclusive bug.
    # Human construction performs no API request; replace before reset.
    return get_env(domain, user_strategy="human", user_model="unused", task_split=split, task_index=task_id)


class Simulator:
    def __init__(self, endpoint, seed, temperature=0, max_tokens=1024):
        self.endpoint, self.seed = endpoint, seed
        self.temperature, self.max_tokens = temperature, max_tokens
        self.messages, self.events = [], []

    def reset(self, instruction=None):
        # Exact upstream llm simulator system prompt; no constructor API call.
        prompt = LLMUserSimulationEnv.build_system_prompt(self, instruction)
        self.messages = [{"role": "system", "content": prompt},
                         {"role": "user", "content": "Hi! How can I help you today?"}]
        return self.generate()

    def step(self, content):
        self.messages.append({"role": "user", "content": content})
        return self.generate()

    def generate(self):
        response = self.endpoint.complete(self.messages, self.events, temperature=self.temperature,
                                          max_tokens=self.max_tokens, seed=self.seed)
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("simulator_truncated")
        if choice.finish_reason != "stop" or not choice.message.content:
            raise RuntimeError("simulator_invalid_response")
        self.messages.append({"role": "assistant", "content": choice.message.content})
        return choice.message.content

    def get_total_cost(self):
        # Upstream interface requires float. Monetary cost is NOT measured here.
        # Do not publish this interface placeholder as actual API cost.
        return 0.0
