prompts = []
results = []
 # take input prompts from the user
for number in range(1, 4):
    prompt = input(f"Enter prompt {number}: ")
    prompts.append(prompt)
# Analyze each prompt and store dictionary in results
for i, prompt in enumerate(prompts, start=1):
    words = prompt.split()
    word_count = len(words)
    char_count = len(prompt)
    analysis = {
        "prompt_number": i,
        "text": prompt,
        "word_count": word_count,
        "char_count": char_count
    }
    results.append(analysis)
# Determine which prompt contains the most words
highest_prompt = max(results, key=lambda x: x["word_count"])
# Print out the detailed summary
print("--- Analysis Results ---")
for res in results:
    print(f"Prompt {res["prompt_number"]}: {res["word_count"]} words | {res["char_count"]} characters")
print("--- Highest Word Count ---")
print(f"Prompt {highest_prompt["prompt_number"]} has the most words ({highest_prompt["word_count"]} words):")
print(f"{highest_prompt["text"]}")