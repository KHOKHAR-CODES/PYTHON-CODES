while True:
    topic = input("Enter a topic (or type 'exit' to quit): ").strip()
    if topic.lower() == 'exit':
        break
    if not topic:
        print("Please enter a valid topic.")
        continue

    answer = chain.invoke({"topic": topic, "points": 3})

    print("\nAnswer:", answer)
    print()
