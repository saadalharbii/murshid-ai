"""Test queries for MurshidAI system"""
import requests
import json

API_URL = "http://127.0.0.1:8000/api/query"

def test_query(question, lang=""):
    print("=" * 80)
    print(f"Question ({lang}): {question}")
    print("=" * 80)

    response = requests.post(API_URL, json={"question": question})

    if response.status_code == 200:
        data = response.json()
        print(f"\nLanguage Detected: {data['language']}")
        print(f"Query Time: {data['query_time']:.2f}s")
        print(f"Sources Found: {len(data['sources'])}")
        print(f"\nAnswer:\n{data['answer']}")

        if data['sources']:
            print(f"\n--- Top 3 Sources (similarity scores) ---")
            for i, source in enumerate(data['sources'][:3], 1):
                score = source.get('similarity_score', 0)
                content_preview = source['content'][:100].replace('\n', ' ')
                print(f"{i}. Score: {score:.3f} - {content_preview}...")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

    print("\n")

# Test queries
test_query("ما هي أفضل الجامعات في لندن؟", "Arabic")
test_query("How do I apply for a scholarship?", "English")
test_query("كيف أجدد تأشيرتي؟", "Arabic")
test_query("What are the best cities to study in the UK?", "English")
