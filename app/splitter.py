import re

def split_preserve_code(text: str, max_tokens=1000, overlap=200):
    blocks = re.split(r'(```[\s\S]*?```)', text)
    chunks = []
    buf = ""
    buf_words = 0
    for b in blocks:
        wc = len(b.split())
        if buf_words + wc > max_tokens and buf:
            chunks.append(buf.strip())
            keep_words = overlap
            tail = " ".join(buf.split()[-keep_words:]) if keep_words < buf_words else buf
            buf = tail + "\n\n" + b
            buf_words = len(buf.split())
        else:
            buf += "\n\n" + b
            buf_words += wc
    if buf.strip():
        chunks.append(buf.strip())
    return chunks