import re
import time
from pathlib import Path
from typing import Dict, List

import duckdb
import pyarrow as pa
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

def iter_html_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for ext in ("*.html", "*.htm"):
        files.extend(root.rglob(ext))
    return files


def build_html_index(html_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in iter_html_files(html_root):
        digits = re.findall(r"\d+", p.name)
        for d in digits:
            if d not in index:
                index[d] = p
    return index


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    target = soup.find("article") or soup.find("main") or soup.body or soup

    text = target.get_text("\n", strip=True)

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def read_html_text(path: Path) -> str:
    try:
        html = path.read_text(encoding="utf-8", errors="strict")
    except Exception:
        html = path.read_text(encoding="utf-8", errors="replace")
    return html_to_text(html)


def main():
    csv_path = Path("posts_summary.csv").resolve()
    html_root = Path("posts").resolve()
    out_path = Path("posts.lance").resolve()
    published_only = True

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not html_root.exists():
        raise FileNotFoundError(f"HTML root not found: {html_root}")

    con = duckdb.connect()
    con.execute("INSTALL lance FROM community")
    con.execute("LOAD lance")

    csv_path_str = str(csv_path).replace("'", "''")
    con.execute(f"CREATE TABLE posts AS SELECT * FROM read_csv_auto('{csv_path_str}')")

    columns = [row[0] for row in con.execute("DESCRIBE posts").fetchall()]
    
    col_map = {
        "post_id": "post_id",
        "title": "title",
        "subtitle": "sub_title",
        "sub_title": "sub_title",
        "post_date": "post_date",
        "date": "post_date",
        "is_published": "is_published",
        "published": "is_published",
    }

    select_parts = []
    for col in columns:
        lc = col.strip().lower()
        if lc in col_map:
            normalized = col_map[lc]
            if col != normalized:
                col_escaped = f'"{col}"' if ' ' in col or '-' in col else col
                select_parts.append(f"{col_escaped} AS {normalized}")
            else:
                select_parts.append(col)
        else:
            select_parts.append(col)

    con.execute(f"CREATE VIEW posts_norm AS SELECT {', '.join(select_parts)} FROM posts")

    if published_only:
        con.execute("""
            CREATE VIEW posts_filtered AS 
            SELECT * FROM posts_norm 
            WHERE is_published = true 
               OR LOWER(CAST(is_published AS VARCHAR)) = 'true'
        """)
    else:
        con.execute("CREATE VIEW posts_filtered AS SELECT * FROM posts_norm")

    con.execute("""
        CREATE VIEW posts_with_ids AS
        SELECT 
            *,
            CAST(regexp_extract(post_id, '^(\\d+)', 1) AS VARCHAR) AS numeric_id
        FROM posts_filtered
    """)

    rows = con.execute("SELECT * FROM posts_with_ids").fetchall()
    columns = [desc[0] for desc in con.execute("SELECT * FROM posts_with_ids LIMIT 0").description]
    
    data = [dict(zip(columns, row)) for row in rows]

    html_index = build_html_index(html_root)

    blog_texts: List[str] = []
    
    model = SentenceTransformer('all-MiniLM-L6-v2')

    for row in data:
        nid = row.get("numeric_id")
        text = ""

        if nid and nid in html_index:
            chosen = html_index[nid]
            try:
                text = read_html_text(chosen)
            except Exception:
                text = ""
        
        blog_texts.append(text)
    
    combined_texts = []
    for i, row in enumerate(data):
        title = str(row.get("title", "") or "")
        subtitle = str(row.get("sub_title", "") or "")
        blog_text = blog_texts[i]
        combined = f"{title} {subtitle} {blog_text}".strip()
        combined_texts.append(combined if combined else "")
    
    print("Generating embeddings...")
    start_time = time.time()
    embeddings = model.encode(combined_texts, show_progress_bar=True, convert_to_numpy=True)
    embedding_time = time.time() - start_time
    print(f"Embedding generation took {embedding_time:.2f} seconds")

    post_ids = [str(row.get("post_id", "") or "") for row in data]
    titles = [str(row.get("title", "") or "") for row in data]
    sub_titles = [str(row.get("sub_title", "") or "") for row in data]
    post_dates = [str(row.get("post_date", "") or "") for row in data]
    
    embedding_lists = [emb.tolist() for emb in embeddings]
    
    schema = pa.schema([
        ('post_id', pa.string()),
        ('title', pa.string()),
        ('sub_title', pa.string()),
        ('post_date', pa.string()),
        ('blog_text', pa.string()),
        ('embedding', pa.list_(pa.float32())),
    ])
    
    table = pa.table({
        'post_id': post_ids,
        'title': titles,
        'sub_title': sub_titles,
        'post_date': post_dates,
        'blog_text': blog_texts,
        'embedding': embedding_lists,
    }, schema=schema)
    
    con.register('posts_with_embeddings', table)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path_str = str(out_path).replace("'", "''")
    
    print("Writing to Lance dataset...")
    start_time = time.time()
    con.execute(f"""
        COPY (
            SELECT post_id, title, sub_title, post_date, blog_text, embedding
            FROM posts_with_embeddings
        ) TO '{out_path_str}' (FORMAT lance, mode 'overwrite')
    """)
    lance_time = time.time() - start_time
    print(f"Writing to Lance took {lance_time:.2f} seconds")
    
    con.close()


if __name__ == "__main__":
    main()
