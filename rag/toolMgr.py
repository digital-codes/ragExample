import json
from typing import Any, Callable, Dict, List, Optional


class ToolChatSession:
    """
    A lightweight session manager for OpenAI chat with automatic tool handling, with optional logging.
    """
    def __init__(
        self,
        client: Any,
        model: str,
        tools: List[Dict[str, Any]],
        function_map: Dict[str, Callable[..., str]],
        provider: str = "openai",
        logging: bool = False
    ):
        self.client = client
        self.model = model
        self.tools = tools
        self.function_map = function_map
        self.provider = provider
        self.logging = logging
        self.messages: List[Dict[str, Any]] = []
        self.total_tokens: int = 0

    def reset(self) -> None:
        """Clear the conversation history and reset token count."""
        self.messages = []
        self.total_tokens = 0

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_tool_response(self, call_id: str, content: str) -> None:
        # Append the tool's response so the model can see it
        self.messages.append({
            "tool_call_id": call_id,
            "role": "tool",
            "content": content,
        })

    def _model_call(self) -> Any:
        """Invoke the model with the current messages and tools."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto",
        )

    def run(
        self,
        query: str,
        documents: Optional[List[str]] = None,
    ) -> str:
        """
        Run a full chat session: send the query, handle any tool calls, and return the final answer.

        Exits immediately if the query is 'quit' or 'exit'.
        If logging is enabled, prints iteration details and total tokens.
        """
        if query.strip().lower() in {"quit", "exit"}:
            print("Session terminated by user.")
            if self.logging:
                print(f"Total tokens used: {self.total_tokens}")
            return "EXIT"

        # Reset state
        self.reset()

        # Initialize with user query and documents
        self.add_user_message(query)
        if documents:
            docs_content = "documents:\n" + "\n".join(documents)
            self.add_user_message(docs_content)

        iteration = 0
        while True:
            iteration += 1
            response = self._model_call()
            # Track and accumulate tokens
            tokens = getattr(response.usage, 'total_tokens', 0)
            self.total_tokens += tokens

            if self.logging:
                print(f"--- Iteration {iteration} ---")
                print("Messages sent to model:")
                for m in self.messages:
                    print(f"  {m['role']}: {m.get('content', '')}")
                print(f"Model returned {tokens} tokens (total so far: {self.total_tokens})")

            choice = response.choices[0]
            finish_reason = choice.finish_reason

            # If no tools were called, return the content
            if finish_reason != "tool_calls":
                if self.logging:
                    print("Final response content:")
                    print(choice.message.content)
                query = input("Query: ")
                if query.strip().lower() in {"quit", "exit"}:
                    print("Session terminated by user.")
                    if self.logging:
                        print(f"Total tokens used: {self.total_tokens}")
                    return "EXIT"
                else:
                    self.add_user_message(query)
                    continue
                #return choice.message.content

            # Otherwise, handle each tool call
            tool_calls = choice.message.tool_calls
            for call in tool_calls:
                fname = call.function.name
                args = json.loads(call.function.arguments)

                if self.logging:
                    print(f"Tool call: {fname} with args {args}")

                if fname not in self.function_map:
                    raise ValueError(f"No handler for function {fname}")

                # Execute the tool function
                tool_response = self.function_map[fname](**args)

                # Append tool response and continue the loop
                self.add_tool_response(call.id, tool_response)

# Example instantiation with default model and GetColorRank tool
if __name__ == "__main__":
    # Define your OpenAI client instance
    from openai import OpenAI
    import ragConfig as cfg

    url = cfg.localllama["lngUrl"].split("/chat")[0]
    model = "granite-3.3-2b-instruct" 
    key = "1234"
    client = OpenAI(base_url=url, api_key=key)
    
    # client = OpenAI()

    # Define the tool list
    tools = [
        {
            "type": "function",
            "function": {
                "description": "Returns best color from list of options.",
                "name": "GetColorRank",
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "colorSet": {
                            "type": "string",
                            "description": "Comma separated list of colors to check for. Returns best",
                        }
                    },
                    "required": ["colorSet"],
                },
            }
        }
    ]

    # Map function names to their callables
    def get_color_rank(colorSet: str) -> str:
        """ "Returns best color from options."""
        print("Calling getColorRank client side.")
        options = colorSet.split(",")
        #return json.dumps({"Best": options[-1]})
        return ":".join(["Best",options[-1]])


    function_map = {
        "GetColorRank": get_color_rank,
    }

    # Create session with default model
    session = ToolChatSession(
        client=client,
        model=model,
        tools=tools,
        function_map=function_map,
        logging = True
    )

    # Run a query
    query = "need best color. Use only data from documents and tools if required."

    documents = ["document_1:\nblue is bad", "document_2:\ngreen is not good","document_3:\nyellow is unclear"]
    while True:
        answer = session.run(query, documents=documents)
        print("Assistant:", answer)
        if answer == "EXIT":
            break
        query = input("Query: ")

