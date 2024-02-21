from transformers import pipeline

# Load the sentiment analysis pipelines
lxyuan_model = pipeline('sentiment-analysis', model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
finbert_model = pipeline('sentiment-analysis', model="ProsusAI/finbert")

# Load comments from the file
with open('comments.txt', 'r') as file:
    comments = file.read().splitlines()

# Analyze each comment
for comment in comments:
    lxyuan_result = lxyuan_model(comment)
    finbert_result = finbert_model(comment)
    print(f"Comment: {comment}")
    print(f"Lxyuan Model - Sentiment: {lxyuan_result[0]['label']}, Score: {lxyuan_result[0]['score']}")
    print(f"Finbert Model - Sentiment: {finbert_result[0]['label']}, Score: {finbert_result[0]['score']}\n")