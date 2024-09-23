import openai

# Set your OpenAI API key directly
api_key = "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"

# Initialize OpenAI API client with your API key
openai.api_key = api_key

# Function to generate an answer from GPT-3
def generate_answer(prompt):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=100,
        temperature=0.5
    )
    return response.choices[0].text.strip()

# Main program loop
while True:
    question = input("What is your question? ")
    if question.lower() == "quit":
        break

    # Generate answer from GPT-3
    answer = generate_answer(question)

    # Display the answer
    print("Answer: ", answer)