import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'script.js'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Try to reverse the mojibake rounds (utf-8 bytes interpreted as latin-1, then re-encoded)
for _ in range(4):
    try:
        fixed = text.encode('latin-1').decode('utf-8')
    except Exception:
        break
    if fixed == text:
        break
    text = fixed

# Replace leftover artifacts explicitly
repl = {
    'Ã¢ÂÂ¦': '...',
    'ÃÂ·': '·',
    'Ã¢ÂÂ': '—',
    'Ã¢-Â': '◇',
    'Ã¢ÂÂ¦': '…',
}
for bad, good in repl.items():
    text = text.replace(bad, good)

# Clean any remaining corrupted sequences manually found in file
# Specific leftover patterns from previous grep output
# We'll just replace the exact bytes we saw: 'Ã¢Â¦' etc.
# Since encoding is tricky in bash, use direct unicode points
# The corrupted string seen was literally "Ã¢Â¦" (three chars: A tilde, a circ, currency)
# After decode it should become '…'

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print('Fixed mojibake in', path)
print('Checking for leftover artifacts...')
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
for ch in ['Ã', '¢', '¦', '', 'Â·', 'Â·', 'â€', 'â€¢', 'â€”', 'â€“', 'âˆ', 'â–']:
    if ch in content:
        print('LEFT:', repr(ch), 'at lines:', [i+1 for i in range(len(content.splitlines())) if ch in content.splitlines()[i]])
