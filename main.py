from sentimentanalysis.sentimentanalysis import SentimentAnalysisBidirectionalLSTMTemperature
import torch
import re
import numpy as np
import torchtext.vocab as vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
glove = vocab.GloVe(name='6B', dim=100)

embedding_matrix = torch.load('sentimentanalysis/glove_embeddings.pt')

sentiment_analyzer = SentimentAnalysisBidirectionalLSTMTemperature(
    embedding_dim=100,
    hidden_dim=256,
    n_layers=2,
    dropout=0.5,
    pretrained_embedding=embedding_matrix,
    init_temp=7.0
)

sentiment_analyzer.to(device)
sentiment_analyzer.load_state_dict(torch.load('sentimentanalysis/combined_model_weights.pth', map_location=device))
sentiment_analyzer.eval()


def predict_text(text, model, max_length):
    def preprocess(s):
        # Remove all non-word characters (everything except numbers and letters)
        s = re.sub(r"[^\w\s]", '', s)
        # Replace all runs of whitespaces with one space
        s = re.sub(r"\s+", ' ', s)
        # replace digits with no space
        s = re.sub(r"\d", '', s)

        return s
    words = [preprocess(word) for word in text.lower().split()[:max_length]]
    word_indices = [glove.stoi[word] if word in glove.stoi else 0 for word in words]

    if len(word_indices) < max_length:
        word_indices.extend([0] * (max_length - len(word_indices)))

    inputs = torch.tensor(word_indices).unsqueeze(0).to(device)

    batch_size = inputs.size(0)
    h = model.init_hidden(batch_size, device)
    h = tuple([each.data for each in h])

    model.eval()
    with torch.no_grad():
        output, h = model(inputs, h)
        prediction = torch.softmax(output, dim=1).cpu().numpy()

    label_mapping = {2: 'Positive', 1: 'Neutral', 0: 'Negative'}
    predicted_class = label_mapping[np.argmax(prediction)]
    predicted_probabilities = prediction[0][np.argmax(prediction)]

    return predicted_class, predicted_probabilities

texts = "Sen. Bernie Sanders Probe Reveals Incredibly Dangerous Warehouse Conditions During Amazon Prime Day"

prediction, probability = predict_text(texts, sentiment_analyzer, 15)
print(f"Predicted class: {prediction}, Probability: {probability}")