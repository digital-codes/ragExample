# https://python.langchain.com/docs/how_to/custom_chat_model/
import time
from typing import Any, Dict, Iterator, List, Optional

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag')))

import ragDeployUtils as rag

DEBUG = True

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field
import asyncio

# non streaming
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# streaming
from langchain_core.messages import (
    AIMessageChunk,
    FunctionMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
    ToolMessageChunk,
)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag')))
from ragInstrumentation import measure_execution_time


class ChatLocal(BaseChatModel):
    """A custom chat model that echoes the first `parrot_buffer_length` characters
    of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = ChatLocal(parrot_buffer_length=2, model="bird-brain-001")
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    model_name: str = Field(alias="model") # default granite-3.3-2b-instruct-Q4_K_M
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2
    engine: Optional[Any] = None
    streamId : Optional[int] = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Custom object initialization
        self.engine = rag.Llm(provider="localllama", model=kwargs.get("model", self.model_name))
        print("Local model created with ", self.model_name)

    @measure_execution_time
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        
        # Call the local model engine to generate a response        
        if DEBUG: print("Generate called with ", messages)
        response = self.engine.query(messages[-1].content)    
        content = response[0]
        tokens = response[1]
        if DEBUG: print ("Tokens from local model: ", tokens)
        if DEBUG: print("Content from local model: ", content)

        ct_input_tokens = sum(len(message.content) for message in messages)
        ct_output_tokens = tokens

        message = AIMessage(
            content=content,
            additional_kwargs={},  # Used to add additional payload to the message
            response_metadata={  # Use for response metadata
                "time_in_seconds": 3,
                "model_name": self.model_name,
            },
            usage_metadata={
                "input_tokens": ct_input_tokens,
                "output_tokens": ct_output_tokens,
                "total_tokens": ct_input_tokens + ct_output_tokens,
            },
        )
        ##

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @measure_execution_time
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        This method should be implemented if the model can generate output
        in a streaming fashion. If the model does not support streaming,
        do not implement it. In that case streaming requests will be automatically
        handled by the _generate method.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        
        if DEBUG: print("Stream called with ", messages)
        while True:
            response = self.engine.queryStream(messages[-1].content, id=self.streamId)
            if DEBUG: print("Response from local model: ", response)
            
            self.streamId = response[0]

            content = response[1].decode('utf-8') if isinstance(response[1], bytes) else response[1]
            stop_detected = response[2]
            
            if stop_detected:
                self.streamId = None
                break
            
            tokens = 1
            ct_input_tokens = sum(len(message.content) for message in messages)

            usage_metadata = UsageMetadata(
                {
                    "input_tokens": ct_input_tokens,
                    "output_tokens": tokens,
                    "total_tokens": ct_input_tokens + tokens,
                }
            )

            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=content, usage_metadata=usage_metadata)
            )

            if run_manager:
                run_manager.on_llm_new_token(content, chunk=chunk)

            yield chunk

        # Final chunk with response metadata
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                response_metadata={"time_in_sec": 3, "model_name": self.model_name},
            )
        )
        if run_manager:
            run_manager.on_llm_new_token("", chunk=chunk)
        yield chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }

        
async def main():
    # Example usage
    model = ChatLocal(model="granite-3.3-2b-instruct-Q4_K_M")
    result = model.invoke([HumanMessage(content="hello")])
    print(result)  # Should print the echoed response
    
    result = model.invoke([HumanMessage(content="Was ist ein Llama")])
    print(result)  # Should print the echoed response

    for chunk in model.stream("cat"):
        print(chunk.content, end="|", flush=True)


    result = model.batch(["hello", "goodbye"])
    print(result)  # Should print the echoed responses
    
    
    # async for chunk in model.astream("cat"):
    #     print(chunk.content, end="|")
    
    # async for event in model.astream_events("cat", version="v1"):
    #     print(event)
               
        
if __name__ == "__main__":
    asyncio.run(main())
