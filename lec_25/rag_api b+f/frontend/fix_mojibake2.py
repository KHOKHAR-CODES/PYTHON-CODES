"""Fix double-mojibake (utf-8 -> latin-1 -> utf-8) in frontend/script.js.

The file was corrupted by encoding the same utf-8 bytes as latin-1 twice.
Example: '◇' (U+25C7) became 'Ã¢-Â‡' and then 'â-‡'.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else '../frontend/script.js'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Undo up to 3 rounds of latin-1 mis-decoding of utf-8 bytes.
for _ in range(3):
    try:
        fixed = text.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        break
    if fixed == text:
        break
    text = fixed

# Any leftovers that survived the generic pass (e.g. mixed sequences).
repl = {
    'â\u0080\u0094': '—',   # em dash
    'â\u0080¦': '…',   # ellipsis
    'â\u0080§': '◇',   # diamond
    'Â·': '·',         # middle dot
    'â\u0096\u0087': '◇',
    'â\u0080\u0093': '–',   # en dash
}
for bad, good in repl.items():
    text = text.replace(bad, good)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

# Report any remaining mojibake markers.
leftovers = sorted({c for c in text if c in 'âÂâ\u0080Ã'})
print('Fixed', path)
if leftovers:
    print('LEFTOVER MARKERS:', [hex(ord(c)) for c in leftovers])
else:
    print('No leftover mojibake markers.')
