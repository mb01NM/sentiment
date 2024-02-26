from transformers import pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from time import time

# Load the sentiment analysis pipelines
lxyuan_model = pipeline('sentiment-analysis', model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
finbert_model = pipeline('sentiment-analysis', model="ProsusAI/finbert")

# Load comments from the file
with open('comments.txt', 'r') as file:
    comments = file.read().splitlines()

# Load actual sentiments from the file
with open('sentiments.txt', 'r') as file:
    actual_sentiments = file.read().splitlines()

lxyuan_predictions = []
finbert_predictions = []

start_time = time()
for comment in comments:
    lxyuan_result = lxyuan_model(comment)
    lxyuan_predictions.append(lxyuan_result[0]['label'])
end_time = time()
lxyuan_time = end_time - start_time

start_time = time()
for comment in comments:
    finbert_result = finbert_model(comment)
    finbert_predictions.append(finbert_result[0]['label'])
end_time = time()
finbert_time = end_time - start_time

# Calculate metrics
lxyuan_accuracy = accuracy_score(actual_sentiments, lxyuan_predictions)
finbert_accuracy = accuracy_score(actual_sentiments, finbert_predictions)

lxyuan_f1 = f1_score(actual_sentiments, lxyuan_predictions, average='weighted')
finbert_f1 = f1_score(actual_sentiments, finbert_predictions, average='weighted')

lxyuan_cm = confusion_matrix(actual_sentiments, lxyuan_predictions, labels=["positive", "neutral", "negative"])
finbert_cm = confusion_matrix(actual_sentiments, finbert_predictions, labels=["positive", "neutral", "negative"])

# Print results
print(f"Lxyuan Model - Accuracy: {lxyuan_accuracy}, F1 Score: {lxyuan_f1}, Time: {round(lxyuan_time, 2)} seconds")
print(f"Finbert Model - Accuracy: {finbert_accuracy}, F1 Score: {finbert_f1}, Time: {round(finbert_time, 2)} seconds")
print(f"Lxyuan Model - Confusion Matrix: \n{lxyuan_cm}")
print(f"Finbert Model - Confusion Matrix: \n{finbert_cm}")