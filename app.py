import streamlit as st
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

def answer_question(context, question):
    # Load the trained model and tokenizer
    model = BertForQuestionAnswering.from_pretrained('trained_model')
    tokenizer = BertTokenizer.from_pretrained('trained_model')

    # Tokenize question and context
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted start and end positions
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Find the tokens with the highest probability
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)

    # Get the predicted answer span
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    answer = tokenizer.convert_tokens_to_string(tokens[start_index:end_index+1])

    return answer

def main():
    st.title("Question Answering Demo")

    # Input question and context
    question = st.text_input("Enter your question:")
    context = st.text_area("Enter the context:")

    if st.button("Get Answer"):
        if context and question:
            answer = answer_question(context, question)
            st.success("Predicted Answer: {}".format(answer))
        else:
            st.warning("Please enter both question and context.")

if __name__ == "__main__":
    main()
