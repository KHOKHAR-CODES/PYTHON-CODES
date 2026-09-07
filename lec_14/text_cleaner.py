import re
def clean_text(text):

    if text is None:
        return ""
    # remove extra spaces

    text = re.sub(
        r"\s+",
        " ",
        text
    )
    # remove leading/trailing spaces

    text = text.strip()

    return text