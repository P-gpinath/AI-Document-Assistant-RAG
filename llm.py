
import ollama


# Send a prompt to the Llama model
response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "user",
            "content": "Explain Python functions in simple English."
        }
    ]
)


# Display the AI response
print("\n========== AI RESPONSE ==========")

print(response["message"]["content"])
