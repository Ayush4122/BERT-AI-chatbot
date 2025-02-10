import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel
from transformers import AutoModelForSequenceClassification
import json
import random
import numpy as np

class AdvancedFitnessChatbot:
    def __init__(self):
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Question answering model
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        
        # Intent classification model
        self.intent_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)
        
        # Comprehensive knowledge base
        self.knowledge_base = self.load_comprehensive_knowledge()
        
        # Conversation patterns
        self.greeting_patterns = [
            "Hi there! I'm your fitness expert AI. How can I help you today?",
            "Hello! Ready to discuss fitness, nutrition, or exercise?",
            "Hey! Welcome to your personal fitness assistant.",
            "Greetings! What fitness goals can I help you with today?"
        ]
        
        # Miscellaneous response templates
        self.miscellaneous_responses = {
            "thanks": [
                "You're welcome! Always happy to help.",
                "Glad I could assist you.",
                "My pleasure! Fitness is my passion."
            ],
            "bye": [
                "Stay fit and healthy! Goodbye.",
                "Take care and keep moving!",
                "Wishing you success in your fitness journey!"
            ]
        }

    def load_comprehensive_knowledge(self):
        """Load an extensive knowledge base covering multiple fitness domains"""
        return {
            "general_fitness": {
                "workout_types": [
                    "Strength training builds muscle and increases metabolism.",
                    "Cardiovascular exercise improves heart health and endurance.",
                    "Flexibility training prevents injuries and improves mobility.",
                    "High-Intensity Interval Training (HIIT) burns fat efficiently.",
                    "Bodyweight exercises can be done anywhere without equipment."
                ],
                "fitness_goals": [
                    "Weight loss requires calorie deficit and consistent exercise.",
                    "Muscle gain needs progressive overload and proper nutrition.",
                    "Endurance training involves gradually increasing workout intensity.",
                    "Body recomposition combines fat loss and muscle gain strategies."
                ]
            },
            "nutrition": {
                "diet_principles": [
                    "Balanced macronutrients are crucial for optimal performance.",
                    "Protein intake supports muscle recovery and growth.",
                    "Hydration is key for metabolism and overall health.",
                    "Timing of meals impacts workout performance and recovery."
                ],
                "meal_planning": [
                    "Prepare meals in advance to maintain consistent nutrition.",
                    "Include variety to ensure comprehensive nutrient intake.",
                    "Adjust calorie intake based on activity level and goals."
                ]
            },
            "exercise_science": {
                "muscle_groups": [
                    "Compound exercises engage multiple muscle groups simultaneously.",
                    "Isolation exercises target specific muscle development.",
                    "Rest and recovery are essential for muscle growth."
                ],
                "injury_prevention": [
                    "Proper warm-up reduces injury risk.",
                    "Maintain correct form during all exercises.",
                    "Listen to your body and avoid overtraining."
                ]
            }
        }

    def classify_intent(self, query):
        """Advanced intent classification"""
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        intent_labels = [
            'greeting', 'exercise', 'nutrition', 'recovery', 
            'weight_loss', 'muscle_gain', 'injury', 'general', 
            'thanks', 'goodbye'
        ]
        
        # Simulate intent classification (in production, use a trained model)
        if any(word in query.lower() for word in ['hi', 'hello', 'hey']):
            return 'greeting'
        elif any(word in query.lower() for word in ['thank', 'thanks']):
            return 'thanks'
        elif any(word in query.lower() for word in ['bye', 'goodbye']):
            return 'goodbye'
        
        # Default fallback
        return 'general'

    def generate_response(self, query):
        """Generate comprehensive response"""
        intent = self.classify_intent(query)
        
        # Handle specific intent responses
        if intent == 'greeting':
            return random.choice(self.greeting_patterns)
        
        if intent == 'thanks':
            return random.choice(self.miscellaneous_responses['thanks'])
        
        if intent == 'goodbye':
            return random.choice(self.miscellaneous_responses['bye'])
        
        # Extract relevant knowledge
        response_candidates = []
        
        # Search through knowledge base
        for domain in self.knowledge_base.values():
            for category, info_list in domain.items():
                response_candidates.extend(info_list)
        
        # Fallback comprehensive response
        if not response_candidates:
            return "While I couldn't find a specific answer, I recommend consulting a fitness professional for personalized advice."
        
        # Select most relevant response
        selected_response = random.choice(response_candidates)
        
        return f"Based on your query: {selected_response}"

def main():
    st.title("🏋️ Advanced Fitness Expert AI")
    st.sidebar.info("Ask anything about fitness, nutrition, or exercise!")
    
    # Initialize chatbot
    chatbot = AdvancedFitnessChatbot()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("What fitness advice do you need?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display bot response
        with st.chat_message("assistant"):
            response = chatbot.generate_response(prompt)
            st.markdown(response)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
