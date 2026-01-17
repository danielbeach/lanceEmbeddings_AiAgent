import duckdb
from pathlib import Path

def main():
    lance_path = Path("posts.lance").resolve()
    
    if not lance_path.exists():
        raise FileNotFoundError(f"Lance dataset not found: {lance_path}")
    
    con = duckdb.connect()
    con.execute("INSTALL lance FROM community")
    con.execute("LOAD lance")
    
    lance_path_str = str(lance_path).replace("'", "''")
    
    result = con.execute(f"""
        SELECT post_id, title, sub_title, post_date, 
               LENGTH(blog_text) as blog_text_length,
               embedding
        FROM '{lance_path_str}'
        LIMIT 10
    """)
    
    columns = [desc[0] for desc in result.description]
    rows = result.fetchall()
    
    print("First 10 rows from Lance dataset:")
    print("=" * 120)
    
    for i, row in enumerate(rows):
        print(f"\nRow {i+1}:")
        for j, col in enumerate(columns):
            val = row[j]
            if col == 'embedding' and isinstance(val, list):
                embedding_str = f"[{val[0]:.4f}, {val[1]:.4f}, {val[2]:.4f}, ..., {val[-1]:.4f}] (length: {len(val)})"
                print(f"  {col}: {embedding_str}")
            else:
                print(f"  {col}: {val}")
    
    print("\n")
    
    count_result = con.execute(f"SELECT COUNT(*) as total_rows FROM '{lance_path_str}'")
    total = count_result.fetchone()[0]
    print(f"Total rows: {total}")
    
    con.close()

if __name__ == "__main__":
    main()
