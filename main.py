from llama_cpp import Llama

# =========================
# Configuration
# =========================

MODEL_PATH = r"D:\A ai projects\llm\ai_models\Qwen3-4B-Q5_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,

    # GPU
    n_gpu_layers=-1,

    # Context window
    n_ctx=8192,

    # Number of CPU threads
    n_threads=8,

    # Batch size
    n_batch=512,

    # Show llama.cpp information
    verbose=True,
)


# =========================
# System prompt
# =========================

SYSTEM_PROMPT = """You are a helpful, intelligent AI assistant.
Answer clearly and accurately.
Be concise unless the user asks for detailed explanations.
"""


# =========================
# Conversation history
# =========================

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]


# =========================
# Chat function
# =========================

def chat(user_message):
    messages.append({
        "role": "user",
        "content": user_message,
    })

    print("\nQwen: ", end="", flush=True)

    full_response = ""

    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.8,
        stream=True,
    )

    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {})
        text = delta.get("content", "")

        if text:
            print(text, end="", flush=True)
            full_response += text

    print("\n")

    messages.append({
        "role": "assistant",
        "content": full_response,
    })


# =========================
# Commands
# =========================

def show_help():
    print("""
Commands:

  /clear    Clear conversation history
  /help     Show this help
  /exit     Exit the chatbot
  /quit     Exit the chatbot
""")


# =========================
# Main loop
# =========================

def main():
    print("=" * 60)
    print("Qwen3-4B Local Chatbot")
    print("GPU accelerated with llama.cpp")
    print("=" * 60)

    print("\nType /help for commands.")
    print("Type /exit to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ("/exit", "/quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/help":
            show_help()
            continue

        if user_input.lower() == "/clear":
            messages.clear()
            messages.append({
                "role": "system",
                "content": SYSTEM_PROMPT,
            })

            print("Conversation cleared.\n")
            continue

        # Normal conversation
        chat(user_input)


if __name__ == "__main__":
    main()