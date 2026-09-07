# Document Registration Program

# --- Collecting basic document information from the user ---
title = input("Enter the document title: ")
author = input("Enter the author name: ")

# Convert page count and file size to numeric types (int and float)
page_count = int(input("Enter the page count: "))
file_size = float(input("Enter the file size (in MB): "))

# Processed status as a Boolean
processed_input = input("Has the document been processed? (yes/no): ")
processed = processed_input.strip().lower() == "yes"

# --- Collecting five keywords into a list (each entered directly, no loop) ---
print("\nEnter 5 keywords describing the document:")
keyword1 = input("Keyword 1: ")
keyword2 = input("Keyword 2: ")
keyword3 = input("Keyword 3: ")
keyword4 = input("Keyword 4: ")
keyword5 = input("Keyword 5: ")
keywords = [keyword1, keyword2, keyword3, keyword4, keyword5]

# --- Supported formats stored as a tuple (fixed, unchangeable set of formats) ---
supported_formats = ("PDF", "DOCX", "TXT")

# --- Unique topics stored as a set (duplicates entered are automatically removed) ---
print("\nEnter 4 topics for the document:")
topic1 = input("Topic 1: ")
topic2 = input("Topic 2: ")
topic3 = input("Topic 3: ")
topic4 = input("Topic 4: ")
unique_topics = {topic1, topic2, topic3, topic4}

# --- Storing all important values inside a dictionary (metadata) ---
metadata = {
    "title": title,
    "author": author,
    "page_count": page_count,
    "file_size": file_size,
    "processed": processed,
    "keywords": keywords,
    "supported_formats": supported_formats,
    "unique_topics": unique_topics
}

# Output Section

# Print the complete document information clearly
print("\n----- DOCUMENT INFORMATION -----")
print(f"Title: {metadata['title']}")
print(f"Author: {metadata['author']}")
print(f"Page Count: {metadata['page_count']}")
print(f"File Size: {metadata['file_size']} MB")
print(f"Processed: {metadata['processed']}")
print(f"Keywords: {metadata['keywords']}")
print(f"Supported Formats: {metadata['supported_formats']}")
print(f"Unique Topics: {metadata['unique_topics']}")

# Print the type of each important value (demonstrates use of type())
print("\n----- DATA TYPES -----")
print("Type of title:", type(title))
print("Type of page_count:", type(page_count))
print("Type of file_size:", type(file_size))
print("Type of keywords list:", type(keywords))
print("Type of unique_topics set:", type(unique_topics))
print("Type of metadata dictionary:", type(metadata))

# Print the number of characters in the title (demonstrates use of len())
print("\n----- COUNTS -----")
print("Number of characters in title:", len(title))

# Print the number of keywords, supported formats, unique topics, and dictionary fields
print("Number of keywords:", len(keywords))
print("Number of supported formats:", len(supported_formats))
print("Number of unique topics:", len(unique_topics))
print("Number of dictionary fields:", len(metadata))
