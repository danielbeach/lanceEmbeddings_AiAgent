import os
from pathlib import Path

import lancedb
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    lance_path = Path("posts.lance").resolve()
    
    if not lance_path.exists():
        raise FileNotFoundError(f"Lance dataset not found: {lance_path}")
    
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    print("Loading Lance dataset...")
    db = lancedb.connect(str(lance_path.parent))
    
    print("Loading embedding model (this may take a moment)...")
    query_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    table = db.open_table(lance_path.stem)
    
    print("Initializing LLM...")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    system_prompt = """Use the following pieces of context from blog posts to answer the question. 
If you don't know the answer based on the context, say that you don't know. Don't make up an answer."""
    
    print("\nChat ready! Ask questions about your blog posts. Type 'quit' or 'exit' to end.\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\nThinking...")
            
            query_embedding = query_embeddings.embed_query(question)
            
            results_df = table.search(query_embedding).limit(5).to_pandas()
            
            context_parts = []
            sources = []
            
            for _, row in results_df.iterrows():
                title = str(row.get('title', 'Unknown'))
                post_id = str(row.get('post_id', 'Unknown'))
                content = str(row.get('blog_text', ''))
                
                context_parts.append(f"Title: {title}\nContent: {content[:1000]}")
                sources.append({'title': title, 'post_id': post_id})
            
            context = "\n\n---\n\n".join(context_parts)
            
            prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            print(f"\nAssistant: {answer}\n")
            
            print("Sources:")
            for i, source in enumerate(sources[:3], 1):
                print(f"  {i}. {source['title']} (post_id: {source['post_id']})")
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
