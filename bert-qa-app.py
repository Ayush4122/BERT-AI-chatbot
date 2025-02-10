import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel
from transformers import AutoModelForSequenceClassification
import numpy as np

class FitnessBERTChatbot:
    def __init__(self):
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Question answering model
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        
        # Intent classification model
        self.intent_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
        
        # Predefined fitness knowledge base
        self.knowledge_base = {
            'exercise': [
                "Proper form is crucial to prevent injuries.",
                "Strength training should include compound and isolation exercises.",
                "Aim for at least 150 minutes of moderate aerobic activity per week."
            ],
            'nutrition': [
                "Balanced diet includes proteins, carbohydrates, and healthy fats.",
                "Hydration is key for optimal performance.",
                "Protein intake should be 1.6-2.2 grams per kg of body weight for strength training."
            ],
            'recovery': [
                "Rest days are essential for muscle repair and growth.",
                "Sleep 7-9 hours per night for optimal recovery.",
                "Active recovery can include light walking or stretching."
            ]
        }

    def classify_intent(self, query):
        """Classify the intent of the user's query"""
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        outputs = self.intent_model(**inputs)
        intent_labels = ['exercise', 'nutrition', 'recovery', 'injury', 'general']
        predicted_intent = intent_labels[torch.argmax(outputs.logits).item()]
        return predicted_intent

    def generate_response(self, query):
        """Generate a response based on intent and query"""
        # Classify intent
        intent = self.classify_intent(query)
        
        # Retrieve relevant knowledge
        relevant_info = self.knowledge_base.get(intent, [])
        
        # Use QA model to extract precise answer
        best_answer = ""
        max_score = 0
        
        for info in relevant_info:
            inputs = self.tokenizer.encode_plus(query, info, return_tensors='pt')
            outputs = self.qa_model(**inputs)
            
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            # Find the most likely answer span
            start_index = torch.argmax(start_scores)
            end_index = torch.argmax(end_scores)
            
            # Calculate overall score
            score = start_scores[0][start_index] + end_scores[0][end_index]
            
            if score > max_score:
                max_score = score
                best_answer = info
        
        # Fallback response if no good match found
        if not best_answer:
            best_answer = "I recommend consulting a fitness professional for personalized advice."
        
        return f"Based on your {intent} query: {best_answer}"

def main():
    st.title("BERT Fitness Expert Chatbot")
    
    # Initialize chatbot
    chatbot = FitnessBERTChatbot()
    
    # Chat input
    user_query = st.text_input("Ask a fitness question:")
    
    if user_query:
        # Generate response
        response = chatbot.generate_response(user_query)
        st.write(response)

if __name__ == "__main__":
    main()
