from transformers import pipeline

# Load the sentiment analysis pipeline
nlp = pipeline('sentiment-analysis', model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

# Load comments from the file
with open('comments.txt', 'r') as file:
    comments = file.read().splitlines()

# Analyze each comment
for comment in comments:
    result = nlp(comment)
    print(f"Comment: {comment}\nSentiment: {result[0]['label']}, Score: {result[0]['score']}\n")