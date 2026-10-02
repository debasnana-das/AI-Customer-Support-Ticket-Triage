import sys
from pathlib import Path
import streamlit as st
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from inference import predict_ticket

st.set_page_config(page_title='Ticket Triage', layout='centered')
st.title('AI Customer Support Ticket Triage')
st.write('Classifies a support ticket into category + urgency and routes it to a queue.')
text = st.text_area('Support ticket', 'I am blocked by a production API error and need help immediately.')
threshold = st.slider('Human-review confidence threshold', 0.0, 1.0, 0.60, 0.05)
if st.button('Classify ticket'):
    result = predict_ticket(text, threshold)
    st.json(result)
    if result['human_review_required']:
        st.warning('Low-confidence prediction: route to human review.')
