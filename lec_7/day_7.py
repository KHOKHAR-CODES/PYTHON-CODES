import re

def main():
	text = input("Enter a text: ")
	words = text.split()
	token_like_units = re.findall(r"\w+|[^\w\s]", text)
	print("original_text:", text)
	print("characters:", len(text))
	print("words from split:", words)
	print("word_count:", len(words))
	print("token_like_units:", token_like_units)
	print("token_count:", len(token_like_units))


if __name__ == "__main__":
	main()
