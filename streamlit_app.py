import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

from train_bert import train

st.title("Social Bias Detection")

# Add a button to trigger the training process
if st.button("Train Model"):
    st.write("Training the model. This may take a while...")
    train()
    st.write("Model training complete!")


# Load the fine-tuned BERT model
model_path = "trained_model"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(model_path)

# Function to predict social bias
def predict_social_bias(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).detach().numpy().tolist()[0]
    return probabilities


# Streamlit UI
st.title("Social Bias Detection")

input_text = st.text_area("Enter text to check for social bias:")
if st.button("Analyze"):
    if input_text:
        probabilities = predict_social_bias(input_text)
        bias_prob = probabilities[1]
        st.write(f"Probability of social bias: {bias_prob:.2%}")

        if bias_prob > 0.5:
            st.warning("This text may contain social bias.")
        else:
            st.success("This text is unlikely to contain social bias.")
    else:
        st.warning("Please enter some text to analyze.")
