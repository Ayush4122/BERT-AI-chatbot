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
        self.intent_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=15)  # Increased labels
        
        # Enhanced knowledge base
        self.knowledge_base = self.load_comprehensive_knowledge()
        
        # Expanded conversation patterns
        self.greeting_patterns = [
            "Hi there! I'm your fitness expert AI. How can I help you today?",
            "Hello! Ready to discuss fitness, nutrition, or exercise?",
            "Hey! Welcome to your personal fitness assistant.",
            "Greetings! What fitness goals can I help you with today?",
            "Welcome! Let's work together on your fitness journey."
        ]
        
        # Enhanced response templates
        self.miscellaneous_responses = {
            "thanks": [
                "You're welcome! Always happy to help.",
                "Glad I could assist you.",
                "My pleasure! Fitness is my passion.",
                "Anytime! Keep up the great work!"
            ],
            "bye": [
                "Stay fit and healthy! Goodbye.",
                "Take care and keep moving!",
                "Wishing you success in your fitness journey!",
                "Keep crushing your fitness goals! Goodbye!"
            ]
        }

        # Load additional QA pairs
        self.qa_pairs = self.load_qa_pairs()

    def load_qa_pairs(self):
        """Load question-answer pairs from training data"""
        qa_data = [
            # Include all QA pairs from the JSON file
            {"question": "What are the benefits of doing deadlifts?",
             "answer": "Deadlifts strengthen the posterior chain, including your lower back, hamstrings, glutes, and traps. They improve overall strength, posture, and grip while also increasing muscle mass and functional fitness."},
            # ... (all other QA pairs from the JSON)
        ]
        return qa_data

    def load_comprehensive_knowledge(self):
        """Load an extensive knowledge base covering multiple fitness domains"""
        return {
            "general_fitness": {
                "workout_types": [
                    "Strength training builds muscle and increases metabolism.",
                    "Cardiovascular exercise improves heart health and endurance.",
                    "Flexibility training prevents injuries and improves mobility.",
                    "High-Intensity Interval Training (HIIT) burns fat efficiently.",
                    "Bodyweight exercises can be done anywhere without equipment.",
                    "Olympic lifting develops explosive power and athletic performance.",
                    "Kettlebell training provides full-body workouts and functional strength."
                ],
                "fitness_goals": [
                    "Weight loss requires calorie deficit and consistent exercise.",
                    "Muscle gain needs progressive overload and proper nutrition.",
                    "Endurance training involves gradually increasing workout intensity.",
                    "Body recomposition combines fat loss and muscle gain strategies.",
                    "Power development requires explosive movements and proper technique.",
                    "Mobility improvement needs consistent stretching and proper form."
                ]
            },
            "nutrition": {
                "diet_principles": [
                    "Balanced macronutrients are crucial for optimal performance.",
                    "Protein intake supports muscle recovery and growth.",
                    "Hydration is key for metabolism and overall health.",
                    "Timing of meals impacts workout performance and recovery.",
                    "Micronutrients play vital roles in energy production and recovery.",
                    "Pre-workout nutrition should focus on easily digestible carbs."
                ],
                "meal_planning": [
                    "Prepare meals in advance to maintain consistent nutrition.",
                    "Include variety to ensure comprehensive nutrient intake.",
                    "Adjust calorie intake based on activity level and goals.",
                    "Time protein intake around workouts for optimal recovery.",
                    "Consider supplements to fill nutritional gaps."
                ]
            },
            "exercise_science": {
                "muscle_groups": [
                    "Compound exercises engage multiple muscle groups simultaneously.",
                    "Isolation exercises target specific muscle development.",
                    "Rest and recovery are essential for muscle growth.",
                    "Different rep ranges target various muscle fiber types.",
                    "Muscle tension time affects hypertrophy response."
                ],
                "injury_prevention": [
                    "Proper warm-up reduces injury risk.",
                    "Maintain correct form during all exercises.",
                    "Listen to your body and avoid overtraining.",
                    "Progressive overload should be implemented gradually.",
                    "Recovery techniques help prevent overuse injuries."
                ],
                "advanced_techniques": [
                    "Time under tension manipulates muscle growth stimulus.",
                    "Drop sets can help break through plateaus.",
                    "Super sets improve workout efficiency.",
                    "Periodization optimizes long-term progress.",
                    "Mind-muscle connection enhances exercise effectiveness."
                ]
            }
        }

    def classify_intent(self, query):
        """Enhanced intent classification"""
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        intent_labels = [
            'greeting', 'exercise', 'nutrition', 'recovery', 
            'weight_loss', 'muscle_gain', 'injury', 'general',
            'thanks', 'goodbye', 'technique', 'supplements',
            'mobility', 'strength', 'cardio'
        ]
        
        # Enhanced intent classification logic
        if any(word in query.lower() for word in ['hi', 'hello', 'hey']):
            return 'greeting'
        elif any(word in query.lower() for word in ['thank', 'thanks']):
            return 'thanks'
        elif any(word in query.lower() for word in ['bye', 'goodbye']):
            return 'goodbye'
        elif any(word in query.lower() for word in ['form', 'technique', 'how to']):
            return 'technique'
        elif any(word in query.lower() for word in ['supplement', 'protein', 'creatine']):
            return 'supplements'
        
        # Default fallback
        return 'general'

    def find_best_answer(self, query):
        """Find the most relevant answer from QA pairs"""
        best_match = None
        highest_similarity = 0
        
        # Simple keyword matching (could be enhanced with more sophisticated NLP)
        query_words = set(query.lower().split())
        
        for qa_pair in self.qa_pairs:
            question_words = set(qa_pair["question"].lower().split())
            similarity = len(query_words.intersection(question_words)) / len(query_words.union(question_words))
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = qa_pair
        
        return best_match["answer"] if best_match and highest_similarity > 0.3 else None

    def generate_response(self, query):
        """Generate comprehensive response with enhanced knowledge"""
        intent = self.classify_intent(query)
        
        # Handle specific intent responses
        if intent == 'greeting':
            return random.choice(self.greeting_patterns)
        
        if intent == 'thanks':
            return random.choice(self.miscellaneous_responses['thanks'])
        
        if intent == 'goodbye':
            return random.choice(self.miscellaneous_responses['bye'])
        
        # Try to find a specific answer from QA pairs
        specific_answer = self.find_best_answer(query)
        if specific_answer:
            return specific_answer
        
        # Extract relevant knowledge if no specific answer found
        response_candidates = []
        
        # Search through enhanced knowledge base
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
