


import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("DeepSeek-R1-Medical-COT")
    model = AutoModelForCausalLM.from_pretrained("DeepSeek-R1-Medical-COT", trust_remote_code=True)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

st.title("🧠 Medical LLM QA (DeepSeek-R1-Medical-COT)")

user_input = st.text_area("Enter your medical question:")

if st.button("Generate Answer"):
    if user_input:
        with st.spinner('Generating...'):
            inputs = tokenizer(user_input, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=256)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("Answer:")
        st.success(answer)
    else:
        st.warning("Please type a question first!")
