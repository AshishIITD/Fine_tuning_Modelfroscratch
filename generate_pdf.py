#!/usr/bin/env python3
import re
import base64
import os
import subprocess
import sys

# Ensure we use the virtual environment's packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'venv_pdf/lib/python3.12/site-packages')))
import markdown

# Paths
WORKSPACE_DIR = "/home/gaian/Fine_tuning_Modelfroscratch"
MD_PATH = "/home/gaian/.gemini/antigravity/brain/c5364b67-c891-4403-985e-7e89389832b7/lora_qlora_deep_dive.md"
HTML_PATH = os.path.join(WORKSPACE_DIR, "lora_qlora_deep_dive.html")
PDF_PATH = os.path.join(WORKSPACE_DIR, "lora_qlora_deep_dive.pdf")

def get_mermaid_url(code):
    """Encodes Mermaid code for the mermaid.ink API."""
    encoded = base64.urlsafe_b64encode(code.strip().encode('utf-8')).decode('utf-8')
    return f"https://mermaid.ink/svg/{encoded}"

def parse_matrix(matrix_content):
    """Converts LaTeX matrix body (elements separated by & and \\) to an HTML table."""
    rows = matrix_content.strip().split(r'\\')
    html_rows = []
    for row in rows:
        if not row.strip():
            continue
        cols = row.split('&')
        html_cols = "".join(f"<td>{col.strip()}</td>" for col in cols)
        html_rows.append(f"<tr>{html_cols}</tr>")
    return f'<table class="matrix">{"".join(html_rows)}</table>'

def replace_matrices(text):
    """Finds all LaTeX bmatrix blocks and replaces them with HTML matrices."""
    def repl(match):
        content = match.group(1)
        return parse_matrix(content)
    return re.sub(r'\\begin{bmatrix}(.*?)\\end{bmatrix}', repl, text, flags=re.DOTALL)

def replace_math(text):
    """Replaces LaTeX equations with beautiful, highly compatible HTML/CSS structures."""
    # 1. Parse and replace matrices first
    text = replace_matrices(text)

    # 2. Order matters: replace longer strings first to prevent partial replacement of substrings!
    replacements = [
        # Block equations
        ("$$W \in \mathbb{R}^{d \times k}$$", '<div class="equation"><i>W</i> &isin; &reals;<sup><i>d</i> &times; <i>k</i></sup></div>'),
        ("$$y = Wx$$", '<div class="equation"><i>y</i> = <i>W</i><i>x</i></div>'),
        ("$$W \in \mathbb{R}^{4096 \times 4096}$$", '<div class="equation"><i>W</i> &isin; &reals;<sup>4096 &times; 4096</sup></div>'),
        ("$$4096 \times 4096 = 16,777,216 \approx 16.8\\text{Million}$$", '<div class="equation">4096 &times; 4096 = 16,777,216 &approx; 16.8 Million</div>'),
        ("$$4096 \times 4096 = 16,777,216 \approx 16.8\text{Million}$$", '<div class="equation">4096 &times; 4096 = 16,777,216 &approx; 16.8 Million</div>'),
        ("$$W \to W'$$", '<div class="equation"><i>W</i> &rarr; <i>W</i>\'</div>'),
        ("$$W' = W + \Delta W$$", '<div class="equation"><i>W</i>\' = <i>W</i> + &Delta;<i>W</i></div>'),
        ("$$\Delta W = BA$$", '<div class="equation">&Delta;<i>W</i> = <i>B</i><i>A</i></div>'),
        ("$$A \in \mathbb{R}^{r \times k}$$", '<div class="equation"><i>A</i> &isin; &reals;<sup><i>r</i> &times; <i>k</i></sup></div>'),
        ("$$B \in \mathbb{R}^{d \times r}$$", '<div class="equation"><i>B</i> &isin; &reals;<sup><i>d</i> &times; <i>r</i></sup></div>'),
        ("$$r \ll \min(d, k)$$", '<div class="equation"><i>r</i> &DoubleLeftArrow; min(<i>d</i>, <i>k</i>)</div>'),
        ("$$y = (W + BA)x$$", '<div class="equation"><i>y</i> = (<i>W</i> + <i>B</i><i>A</i>)<i>x</i></div>'),
        ("$$y = Wx + BAx$$", '<div class="equation"><i>y</i> = <i>W</i><i>x</i> + <i>B</i><i>A</i><i>x</i></div>'),
        ("$$y = Wx + \\frac{\\alpha}{r} BAx$$", '<div class="equation"><i>y</i> = <i>W</i><i>x</i> + <table class="frac"><tr><td class="num">&alpha;</td></tr><tr><td class="den"><i>r</i></td></tr></table> <i>B</i><i>A</i><i>x</i></div>'),
        ("$$BA = 0$$", '<div class="equation"><i>B</i><i>A</i> = 0</div>'),
        ("$$y = Wx + 0 = Wx$$", '<div class="equation"><i>y</i> = <i>W</i><i>x</i> + 0 = <i>W</i><i>x</i></div>'),
        ("$$W_{\\text{final}} = W + \\frac{\\alpha}{r} BA$$", '<div class="equation"><i>W</i><sub>final</sub> = <i>W</i> + <table class="frac"><tr><td class="num">&alpha;</td></tr><tr><td class="den"><i>r</i></td></tr></table> <i>B</i><i>A</i></div>'),
        ("$$q = \\text{round}\\left(\\frac{w}{s}\\right)$$", '<div class="equation"><i>q</i> = round<table class="frac"><tr><td class="num"><i>w</i></td></tr><tr><td class="den"><i>s</i></td></tr></table></div>'),
        ("$$w \\approx s \\cdot q$$", '<div class="equation"><i>w</i> &approx; <i>s</i> &middot; <i>q</i></div>'),
        ("$$y = \\text{Dequantize}(W_q)x + \\frac{\\alpha}{r} BAx$$", '<div class="equation"><i>y</i> = Dequantize(<i>W</i><sub><i>q</i></sub>)<i>x</i> + <table class="frac"><tr><td class="num">&alpha;</td></tr><tr><td class="den"><i>r</i></td></tr></table> <i>B</i><i>A</i><i>x</i></div>'),
        ("$$L = \\frac{1}{2} \\sum (y_{\\text{pred}} - y_{\\text{true}})^2$$", '<div class="equation"><i>L</i> = <table class="frac"><tr><td class="num">1</td></tr><tr><td class="den">2</td></tr></table> &sum; (<i>y</i><sub>pred</sub> - <i>y</i><sub>true</sub>)<sup>2</sup></div>'),
        ("$$L = \\frac{1}{2} \\left( (8 - 10)^2 + (18 - 20)^2 \\right) = \\frac{1}{2} (4 + 4) = 4$$", '<div class="equation"><i>L</i> = <table class="frac"><tr><td class="num">1</td></tr><tr><td class="den">2</td></tr></table> ( (8 - 10)<sup>2</sup> + (18 - 20)<sup>2</sup> ) = <table class="frac"><tr><td class="num">1</td></tr><tr><td class="den">2</td></tr></table> (4 + 4) = 4</div>'),
        ("$$\\frac{\\partial L}{\\partial W_{ij}} = (y_{\\text{pred}, i} - y_{\\text{true}, i}) x_j$$", '<div class="equation"><table class="frac"><tr><td class="num">&part;<i>L</i></td></tr><tr><td class="den">&part;<i>W</i><sub><i>ij</i></sub></td></tr></table> = (<i>y</i><sub>pred, <i>i</i></sub> - <i>y</i><sub>true, <i>i</i></sub>) <i>x</i><sub><i>j</i></sub></div>'),
        ("$$W_{\\text{new}} = W - \\eta \\frac{\\partial L}{\\partial W}$$", '<div class="equation"><i>W</i><sub>new</sub> = <i>W</i> - &eta; <table class="frac"><tr><td class="num">&part;<i>L</i></td></tr><tr><td class="den">&part;<i>W</i></td></tr></table></div>'),
        ("$$W_{\\text{new}} = W + BA = <table class=\"matrix\"><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table> + <table class=\"matrix\"><tr><td>0.2</td><td>0.3</td></tr><tr><td>0.2</td><td>0.3</td></tr></table> = <table class=\"matrix\"><tr><td>1.2</td><td>2.3</td></tr><tr><td>3.2</td><td>4.3</td></tr></table>$$", '<div class="equation"><i>W</i><sub>new</sub> = <i>W</i> + <i>B</i><i>A</i> = <table class="matrix"><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table> + <table class="matrix"><tr><td>0.2</td><td>0.3</td></tr><tr><td>0.2</td><td>0.3</td></tr></table> = <table class="matrix"><tr><td>1.2</td><td>2.3</td></tr><tr><td>3.2</td><td>4.3</td></tr></table></div>'),
        ("$$W_{\\text{new}} = W - 0.05 \\frac{\\partial L}{\\partial W}$$", '<div class="equation"><i>W</i><sub>new</sub> = <i>W</i> - 0.05 <table class="frac"><tr><td class="num">&part;<i>L</i></td></tr><tr><td class="den">&part;<i>W</i></td></tr></table></div>'),

        # Inline equations (longer first!)
        ("$W_{\\text{new}} = W - \\eta \\frac{\\partial L}{\\partial W}$", '<i>W</i><sub>new</sub> = <i>W</i> - &eta; <table class="frac"><tr><td class="num">&part;<i>L</i></td></tr><tr><td class="den">&part;<i>W</i></td></tr></table>'),
        ("$W_{\\text{final}} = W + \\frac{\\alpha}{r} BA$", '<i>W</i><sub>final</sub> = <i>W</i> + <table class="frac"><tr><td class="num">&alpha;</td></tr><tr><td class="den"><i>r</i></td></tr></table> <i>B</i><i>A</i>'),
        ("$y = \\text{Dequantize}(W_q)x + \\frac{\\alpha}{r} BAx$", '<i>y</i> = Dequantize(<i>W</i><sub><i>q</i></sub>)<i>x</i> + <table class="frac"><tr><td class="num">&alpha;</td></tr><tr><td class="den"><i>r</i></td></tr></table> <i>B</i><i>A</i><i>x</i>'),
        ("$W_0 \\in \\mathbb{R}^{d \\times k}$", '<i>W</i><sub>0</sub> &isin; &reals;<sup><i>d</i> &times; <i>k</i></sup>'),
        ("$W \\in \\mathbb{R}^{d \\times k}$", '<i>W</i> &isin; &reals;<sup><i>d</i> &times; <i>k</i></sup>'),
        ("$x \\in \\mathbb{R}^{k}$", '<i>x</i> &isin; &reals;<sup><i>k</i></sup>'),
        ("$\\Delta W = BA = 0$", '&Delta;<i>W</i> = <i>B</i><i>A</i> = 0'),
        ("$\\Delta W = BA$", '&Delta;<i>W</i> = <i>B</i><i>A</i>'),
        ("$A \\in \\mathbb{R}^{r \\times k}$", '<i>A</i> &isin; &reals;<sup><i>r</i> &times; <i>k</i></sup>'),
        ("$B \\in \\mathbb{R}^{d \\times r}$", '<i>B</i> &isin; &reals;<sup><i>d</i> &times; <i>r</i></sup>'),
        ("$r \\ll \\min(d, k)$", '<i>r</i> &DoubleLeftArrow; min(<i>d</i>, <i>k</i>)'),
        ("$r \\in \\{4, 8, 16, 32, 64\\}$", '<i>r</i> &isin; {4, 8, 16, 32, 64}'),
        ("$y = (W + BA)x$", '<i>y</i> = (<i>W</i> + <i>B</i><i>A</i>)<i>x</i>'),
        ("$y = Wx + BAx$", '<i>y</i> = <i>W</i><i>x</i> + <i>B</i><i>A</i><i>x</i>'),
        ("$y = Wx$", '<i>y</i> = <i>W</i><i>x</i>'),
        ("$W \\to W'$", '<i>W</i> &rarr; <i>W</i>\''),
        ("$W' = W + \Delta W$", '<i>W</i>\' = <i>W</i> + &Delta;<i>W</i>'),
        ("$\\Delta W \\in \\mathbb{R}^{d \\times k}$", '&Delta;<i>W</i> &isin; &reals;<sup><i>d</i> &times; <i>k</i></sup>'),
        ("$\\mathcal{N}(0, \\sigma^2)$", '&Nscr;(0, &sigma;<sup>2</sup>)'),
        ("$\\mathcal{N}(0, 1)$", '&Nscr;(0, 1)'),
        ("$\\Delta W$", '&Delta;<i>W</i>'),
        ("$W_{\\text{dequant}} = W_q \\cdot s$", '<i>W</i><sub>dequant</sub> = <i>W</i><sub><i>q</i></sub> &middot; <i>s</i>'),
        ("$y_{\\text{pred}} = \\begin{bmatrix} 8 \\\\ 18 \\end{bmatrix}$", '<i>y</i><sub>pred</sub> = <table class="matrix"><tr><td>8</td></tr><tr><td>18</td></tr></table>'),
        ("$y_{\\text{new}} = \\begin{bmatrix} 9.3 \\\\ 19.3 \\end{bmatrix}$", '<i>y</i><sub>new</sub> = <table class="matrix"><tr><td>9.3</td></tr><tr><td>19.3</td></tr></table>'),
        ("$x = \\begin{bmatrix} 2 \\\\ 3 \end{bmatrix}$", '<i>x</i> = <table class="matrix"><tr><td>2</td></tr><tr><td>3</td></tr></table>'),
        ("$B = 0$", '<i>B</i> = 0'),
        ("$BA = 0$", '<i>B</i><i>A</i> = 0'),
        ("$\\alpha$", '&alpha;'),
        ("$\\frac{\\alpha}{r}$", '<table class="frac"><tr><td class="num">&alpha;</td></tr><tr><td class="den"><i>r</i></td></tr></table>'),
        ("$\\frac{32}{64} = 0.5$", '<table class="frac"><tr><td class="num">32</td></tr><tr><td class="den">64</td></tr></table> = 0.5'),
        ("$c_1$", '<i>c</i><sub>1</sub>'),
        ("$d$", '<i>d</i>'),
        ("$k$", '<i>k</i>'),
        ("$r$", '<i>r</i>'),
        ("$A$", '<i>A</i>'),
        ("$B$", '<i>B</i>'),
        ("$W$", '<i>W</i>'),
        ("$\\eta = 0.05$", '&eta; = 0.05'),
        ("$\\eta = 1$", '&eta; = 1'),
        ("$s = 0.25$", '<i>s</i> = 0.25'),
        ("$W_q$", '<i>W</i><sub><i>q</i></sub>'),
        ("$q = \\text{round}(w/s)$", '<i>q</i> = round(<i>w</i>/<i>s</i>)'),
        ("$w \\approx s \\cdot q$", '<i>w</i> &approx; <i>s</i> &middot; <i>q</i>')
    ]
    
    for old, new in replacements:
        text = text.replace(old, new)
        
    return text

def process_markdown(md_content):
    """Parses markdown, replacing Mermaid blocks and formatting equations."""
    # Replace Mermaid blocks with SVG images
    def mermaid_replacer(match):
        code = match.group(1)
        url = get_mermaid_url(code)
        return f'\n<div class="mermaid-diagram"><img src="{url}" alt="Mermaid Diagram"></div>\n'

    processed = re.sub(r'```mermaid\n(.*?)\n```', mermaid_replacer, md_content, flags=re.DOTALL)
    
    # Format math equations
    processed = replace_math(processed)
    
    return processed

# Premium CSS and HTML structure
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LoRA and QLoRA Masterclass Deep Dive</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary: #4f46e5;
            --primary-dark: #3730a3;
            --text-main: #1f2937;
            --text-light: #4b5563;
            --bg-main: #ffffff;
            --bg-code: #0f172a;
            --border-color: #e5e7eb;
        }}
        
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: var(--text-main);
            background-color: var(--bg-main);
            margin: 0;
            padding: 0;
        }}
        
        .page {{
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        /* Typography */
        h1, h2, h3, h4 {{
            font-family: 'Outfit', sans-serif;
            color: #111827;
            font-weight: 700;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            page-break-after: avoid;
        }}
        
        h1 {{
            font-size: 24pt;
            border-bottom: 3px solid var(--primary);
            padding-bottom: 8px;
            margin-top: 0;
            color: #1e1b4b;
        }}
        
        h2 {{
            font-size: 17pt;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 6px;
            color: #312e81;
            margin-top: 1.8em;
        }}
        
        h3 {{
            font-size: 12.5pt;
            color: var(--primary-dark);
        }}
        
        p {{
            margin-top: 0;
            margin-bottom: 1em;
            text-align: justify;
        }}
        
        a {{
            color: var(--primary);
            text-decoration: none;
        }}
        
        /* Lists */
        ul, ol {{
            margin-top: 0;
            margin-bottom: 1em;
            padding-left: 20px;
        }}
        
        li {{
            margin-bottom: 0.4em;
        }}
        
        /* Tables */
        table.table-markdown {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5em 0;
            font-size: 10pt;
            page-break-inside: avoid;
        }}
        
        table.table-markdown th, table.table-markdown td {{
            border: 1px solid var(--border-color);
            padding: 10px 12px;
            text-align: left;
        }}
        
        table.table-markdown th {{
            background-color: #f8fafc;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            color: #0f172a;
        }}
        
        table.table-markdown tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        
        /* Code blocks */
        pre {{
            background-color: var(--bg-code);
            color: #f8fafc;
            padding: 14px;
            border-radius: 8px;
            font-family: 'Fira Code', 'Courier New', monospace;
            font-size: 9pt;
            overflow-x: auto;
            margin: 1.2em 0;
            border: 1px solid #1e293b;
            page-break-inside: avoid;
        }}
        
        code {{
            font-family: 'Fira Code', monospace;
            font-size: 9.5pt;
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 2px 5px;
            border-radius: 4px;
        }}
        
        pre code {{
            background-color: transparent;
            color: inherit;
            padding: 0;
            border-radius: 0;
            font-size: inherit;
        }}
        
        /* Mermaid diagrams */
        .mermaid-diagram {{
            text-align: center;
            margin: 2em 0;
            page-break-inside: avoid;
        }}
        
        .mermaid-diagram img {{
            max-width: 100%;
            height: auto;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 15px;
            background-color: #fafafa;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }}
        
        /* Equations */
        .equation {{
            text-align: center;
            font-size: 12pt;
            margin: 1.5em 0;
            padding: 14px;
            background-color: #f8fafc;
            border-radius: 6px;
            border-left: 4px solid var(--primary);
            font-family: 'Outfit', serif;
            color: #0f172a;
            page-break-inside: avoid;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }}
        
        /* CSS Fractions */
        table.frac {{
            display: inline-table;
            vertical-align: middle;
            border-collapse: collapse;
            margin: 0 0.3em;
        }}
        table.frac td {{
            padding: 0;
            text-align: center;
            border: none !important;
            background: transparent !important;
        }}
        table.frac td.num {{
            border-bottom: 1px solid #1f2937 !important;
            font-size: 0.85em;
            padding-bottom: 1px;
            color: inherit;
        }}
        table.frac td.den {{
            font-size: 0.85em;
            padding-top: 1px;
            color: inherit;
        }}
        
        /* CSS Matrices */
        table.matrix {{
            display: inline-table;
            vertical-align: middle;
            border-collapse: collapse;
            margin: 0 0.6em;
            position: relative;
            padding: 0 6px;
            background: transparent !important;
        }}
        table.matrix::before, table.matrix::after {{
            content: "";
            position: absolute;
            top: 0;
            bottom: 0;
            width: 5px;
            border: 1.5px solid #1f2937;
        }}
        table.matrix::before {{
            left: 0;
            border-right: none;
        }}
        table.matrix::after {{
            right: 0;
            border-left: none;
        }}
        table.matrix td {{
            padding: 4px 10px;
            text-align: center;
            border: none !important;
            font-size: 10.5pt;
            font-family: 'Outfit', sans-serif;
            color: #0f172a;
            background: transparent !important;
        }}
        
        /* Page break controls */
        .hr-break {{
            page-break-after: always;
            border: none;
            margin: 0;
        }}
        
        /* Print optimization */
        @media print {{
            @page {{
                size: A4;
                margin: 20mm 15mm 20mm 15mm;
            }}
            body {{
                font-size: 10pt;
                color: #000;
                background-color: #fff;
            }}
            .page {{
                width: 100%;
                padding: 0;
            }}
            pre {{
                border: 1px solid #cbd5e1;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            table.table-markdown th {{
                background-color: #f1f5f9 !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .equation {{
                background-color: #f1f5f9 !important;
                border-left: 4px solid var(--primary) !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
        }}
    </style>
</head>
<body>
    <div class="page">
        {content}
    </div>
</body>
</html>
"""

def main():
    print("Reading markdown file...")
    with open(MD_PATH, 'r') as f:
        md_content = f.read()

    # Pre-process math and Mermaid blocks
    print("Processing mathematical formulas and Mermaid diagrams...")
    processed_content = process_markdown(md_content)

    # Convert Markdown to HTML
    print("Converting Markdown to HTML...")
    # Use standard markdown conversion
    md = markdown.Markdown(extensions=['extra', 'codehilite', 'toc'])
    html_body = md.convert(processed_content)

    # Make sure markdown tables get a class for styling
    html_body = html_body.replace("<table>", '<table class="table-markdown">')

    # Wrap in HTML template
    full_html = HTML_TEMPLATE.format(content=html_body)

    # Write HTML file
    print(f"Writing temporary HTML file to {HTML_PATH}...")
    with open(HTML_PATH, 'w') as f:
        f.write(full_html)

    # Run Headless Chrome to compile PDF
    print(f"Compiling PDF using Google Chrome headless...")
    chrome_cmd = [
        "/usr/bin/google-chrome",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        f"--print-to-pdf={PDF_PATH}",
        HTML_PATH
    ]
    
    try:
        res = subprocess.run(chrome_cmd, capture_output=True, text=True, check=True)
        print("PDF compilation completed successfully!")
        print(f"Saved PDF to: {PDF_PATH}")
        
        # Clean up HTML file
        if os.path.exists(HTML_PATH):
            os.remove(HTML_PATH)
            print("Cleaned up temporary HTML file.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error compiling PDF: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()
