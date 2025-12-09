from kani import ChatMessage
from kani.engines.base import BaseEngine, Completion


class TestEngine(BaseEngine):
    """A mock engine used for testing.

    Each message has a token length equal to its str length, and predict always returns a one-token message.
    """

    max_context_size = 10

    def prompt_len(self, messages, functions=None, **kwargs):
        return sum(len(m.text) for m in messages)

    async def predict(self, messages, functions=None, test_echo=False, **hyperparams) -> Completion:
        """
        :param test_echo: If True, the prediction echoes the last message.
        """
        assert sum(len(m.text or "") for m in messages) <= self.max_context_size
        if test_echo:
            content = messages[-1].text
            return Completion(ChatMessage.assistant(content))
        return Completion(ChatMessage.assistant("a"))
