def preview(text: str, max_chars: int) -> str:
    text = ' '.join(text.split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + '…'
