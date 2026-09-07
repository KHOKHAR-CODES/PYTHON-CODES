# text utlites.py
def clean_text(text):
    """return lowercase text with ouside spaces removed"""
    return text.strip().lower()

def prepare_keyword(keyword):
    """clean a user keywords before searching"""
    return keyword.strip().lower()

def contains_keyword(text, keyword):
    """return True if the keyword is in the text"""
    return keyword in text
    