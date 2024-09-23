from flask import Flask, request, jsonify, render_template
import torch
import torch
import json
from utils import SAM, process_tweet

with open('vocabulary.json', 'r') as json_file:
    vocabulary = json.load(json_file)
model = torch.load('models/sentiment_model_full.pth')

def tweet_to_tensor(X):
    X_final = []
    max_length = 51
    for tweet in X:
        processed_tweet = process_tweet(tweet)
        processed_tensor = []
        
        for word in processed_tweet:
            if word in vocabulary:
                processed_tensor.append(vocabulary[word])
            else:
                processed_tensor.append(vocabulary['_unk_'])
        
        padding_length = max_length - len(processed_tensor)
        processed_tensor = processed_tensor + [0]*padding_length
        X_final.append(processed_tensor)
    
    X_final = torch.tensor(X_final)
    
    return X_final

app = Flask(__name__)



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    data = request.json
    prompt = data.get('input', '')
    
    model.eval()
    tensor = tweet_to_tensor([prompt])
    with torch.no_grad():
        logits = model(tensor)
        prediction = (torch.sigmoid(logits) > 0.5).int().squeeze()
        if prediction.item() == 1:
            result = 'Positive Statement'
        else:
            result = 'Negative Statement'
    
    return jsonify({'result': result})
    

# Run the app
if __name__ == '__main__':
    app.run(debug=True)