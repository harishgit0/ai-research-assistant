from app.services.generation.gemini_provider import GeminiProvider


provider = GeminiProvider()

response = provider.generate(
    "Explain retrieval-augmented generation in exactly two sentences."
)

print("\n--- Gemini Response ---")
print(response)