# Import necessary libraries
import streamlit as st
import openai
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Set OpenAI API key for GPT-3
openai.api_key = "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"

# Load the GPT-2 and GPT-3 models
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")



models = {
    "GPT-2": gpt2_model,
}

# Predict hate speech and social bias using the GPT-2 model
def predict_gpt2(generator, text):
    # Example implementation using GPT-2 text generation for classification (not accurate)
    prompt = f"Hate speech classification: {text}"
    generated_text = generator(prompt, max_length=50, do_sample=True)[0]['generated_text']
    hate_speech_result = 'yes' in generated_text.lower()

    prompt = f"Social bias classification: {text}"
    generated_text = generator(prompt, max_length=50, do_sample=True)[0]['generated_text']
    social_bias_result = 'yes' in generated_text.lower()

    return hate_speech_result, social_bias_result

# Streamlit app
def main():
    st.set_page_config(page_title="Hate Speech and Social Bias Detection", layout="wide")
    st.title("Hate Speech and Social Bias Detection")

    # Model selection
    model_name = st.sidebar.selectbox("Select a model", list(models.keys()))

    # Input text
    user_input = st.text_area("Input text for analysis", "")

    # Process input text
    if st.button("Analyze"):
        if user_input:
            model = models[model_name]

            if model_name == "GPT-2":
                # Hate speech and social bias detection using GPT-2
                hate_speech_result, social_bias_result = predict_gpt2(gpt2_model, gpt2_tokenizer, user_input)

            # Display results
            st.write("### Hate Speech Analysis")
            st.write(f"Hate speech detected: {'Yes' if hate_speech_result else 'No'}")

            st.write("### Social Bias Analysis")
            st.write(f"Social bias detected: {'Yes' if social_bias_result else 'No'}")
        else:
            st.error("Please provide input text for analysis.")

if __name__ == "__main__":
    main()
