import streamlit as st

# Tiêu đề
st.title("Question Answering Demo")

# Input question và context
question = st.text_input("Question:", "What is Beyoncé's hometown?")
context = st.text_area("Context:", "Beyoncé was born and raised in Houston, Texas.")

# Button để thực hiện dự đoán
if st.button("Predict"):
    # Tiến hành dự đoán và hiển thị kết quả
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    answer = tokenizer.convert_tokens_to_string(tokens[start_index:end_index+1])
    st.success("Predicted Answer: " + answer)
