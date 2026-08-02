"""
build.py — Tạo file HTML tự chứa (standalone) cho Medical NER Labeler
Chạy: python build.py
Output: dist/Medical_NER_Labeler.html (single file, hoàn toàn offline)
"""

import re, base64, os, urllib.request, urllib.error

SRC = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(SRC, "dist")
os.makedirs(DIST, exist_ok=True)

# ---------- helpers ----------
def read(path, encoding="utf-8"):
    with open(path, encoding=encoding) as f:
        return f.read()

def fetch_url(url, retries=3):
    headers = {"User-Agent": "Mozilla/5.0"}
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.read()
        except Exception as e:
            if i == retries - 1:
                raise
    return b""

def fetch_font_as_b64(url):
    data = fetch_url(url)
    # Detect mime
    if url.endswith(".woff2"):
        mime = "font/woff2"
    elif url.endswith(".woff"):
        mime = "font/woff"
    elif url.endswith(".ttf"):
        mime = "font/ttf"
    else:
        mime = "font/woff2"
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"

def inline_google_fonts(css_url):
    """Download @font-face CSS, then replace each url(...) with base64."""
    print(f"  Fetching font CSS: {css_url}")
    # Use a desktop UA so we get woff2
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    req = urllib.request.Request(css_url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=15) as r:
        css_text = r.read().decode("utf-8")

    # Find all font urls
    font_urls = re.findall(r"url\((https://[^)]+)\)", css_text)
    seen = {}
    for fu in font_urls:
        if fu in seen:
            continue
        print(f"    Embedding font: {fu.split('/')[-1]}")
        try:
            b64_uri = fetch_font_as_b64(fu)
            seen[fu] = b64_uri
        except Exception as e:
            print(f"    WARNING: could not fetch {fu}: {e}")
            seen[fu] = fu

    for orig, replacement in seen.items():
        css_text = css_text.replace(f"url({orig})", f"url({replacement})")

    return css_text

# ---------- main ----------
print("=== Building standalone Medical NER Labeler ===\n")

html = read(os.path.join(SRC, "index.html"))
css  = read(os.path.join(SRC, "style.css"))
js   = read(os.path.join(SRC, "app.js"))

# 1. Inline Google Fonts
GFONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@300;400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500"
    "&display=swap"
)

try:
    print("Step 1: Downloading & embedding Google Fonts...")
    font_css = inline_google_fonts(GFONTS_URL)
    font_block = f"<style id='gfonts'>\n{font_css}\n</style>"
    # Remove the external link tag from HTML
    html = re.sub(
        r'<link[^>]*fonts\.googleapis\.com[^>]*>',
        '',
        html
    )
    html = re.sub(
        r'<link[^>]*fonts\.gstatic\.com[^>]*>',
        '',
        html
    )
    # Also remove preconnect links for gstatic/googleapis
    html = re.sub(r'<link[^>]*preconnect[^>]*googleapis[^>]*>', '', html)
    html = re.sub(r'<link[^>]*preconnect[^>]*gstatic[^>]*>', '', html)
    print("  Google Fonts embedded successfully.\n")
except Exception as e:
    print(f"  WARNING: Could not embed fonts ({e}). Will use system fonts.\n")
    font_block = "<style id='gfonts'>/* Google Fonts not available offline */</style>"

# 2. Inline CSS (remove external stylesheet link, embed inline)
print("Step 2: Inlining CSS...")
css_block = f"<style id='app-css'>\n{css}\n</style>"
html = re.sub(r'<link[^>]*stylesheet[^>]*style\.css[^>]*>', '', html)
print("  Done.\n")

# 3. Inline JS (remove external script tag, embed inline)
print("Step 3: Inlining JavaScript...")
js_block = f"<script id='app-js'>\n{js}\n</script>"
html = re.sub(r'<script[^>]*src=["\']app\.js["\'][^>]*></script>', '', html)
print("  Done.\n")

# 4. Inject into <head> and before </body>
# Insert font + css into <head>
html = html.replace("</head>", f"\n{font_block}\n{css_block}\n</head>")
# Insert js before </body>
html = html.replace("</body>", f"\n{js_block}\n</body>")

# 5. Clean up extra blank lines
html = re.sub(r'\n{3,}', '\n\n', html)

# 6. Write output
out_path = os.path.join(DIST, "Medical_NER_Labeler.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize(out_path) / 1024
print(f"=== Build complete! ===")
print(f"Output : {out_path}")
print(f"Size   : {size_kb:.1f} KB")
print(f"\nChỉ cần copy file '{os.path.basename(out_path)}' sang máy mới,")
print("mở bằng bất kỳ trình duyệt nào (Chrome, Edge, Firefox...) là dùng được.")
print("Không cần cài đặt gì thêm.\n")
