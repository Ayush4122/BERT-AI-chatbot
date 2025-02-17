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
             {
      "question": "How often should I do cardio to lose fat?",
      "answer": "For fat loss, aim for 3-5 sessions of cardio per week, each lasting 30-45 minutes. Combine steady-state cardio with high-intensity interval training (HIIT) for the best results."
    },
    {
      "question": "What's the best way to build muscle?",
      "answer": "To build muscle, focus on progressive overload with compound exercises like squats, deadlifts, bench presses, and rows. Train each muscle group 2-3 times per week, prioritize protein intake, and ensure adequate rest and recovery."
    },
    {
      "question": "What should I eat before a workout?",
      "answer": "Eat a balanced meal 1-3 hours before your workout that includes complex carbs, protein, and some fats. For example, oatmeal with a scoop of protein powder or a chicken sandwich with whole grain bread."
    },
    {
      "question": "How much protein should I consume daily?",
      "answer": "For muscle building, consume about 1.6-2.2 grams of protein per kilogram of body weight per day. For general health and maintenance, aim for 1.0-1.2 grams per kilogram."
    },
    {
      "question": "What's the proper form for a bench press?",
      "answer": "Keep your feet flat on the floor, arch your lower back slightly, and keep your shoulder blades retracted. Lower the bar to your mid-chest, keep your elbows at a 45-degree angle, and push the bar up in a controlled motion."
    },
    {
      "question": "Can I lose fat and build muscle at the same time?",
      "answer": "Yes, but it requires a careful balance of nutrition and training. Focus on strength training, maintain a slight calorie deficit, and ensure sufficient protein intake to support muscle growth while losing fat."
    },
    {
      "question": "What are the benefits of HIIT?",
      "answer": "HIIT improves cardiovascular fitness, burns fat efficiently, and preserves muscle mass. It's time-efficient and boosts metabolism for hours after the workout due to the EPOC (Excess Post-exercise Oxygen Consumption) effect."
    },
    {
      "question": "How can I improve my squat depth?",
      "answer": "To improve squat depth, work on ankle mobility, hip flexibility, and core stability. Incorporate exercises like goblet squats, hip flexor stretches, and ankle dorsiflexion drills."
    },
    {
      "question": "How do I prevent muscle soreness after a workout?",
      "answer": "To reduce muscle soreness, stay hydrated, perform a proper warm-up and cool-down, and include stretching. Active recovery, such as light walking or swimming, can also help alleviate soreness."
    },
    {
      "question": "Is it necessary to take supplements for muscle growth?",
      "answer": "Supplements are not mandatory for muscle growth, but they can support your diet. Protein powder, creatine, and branched-chain amino acids (BCAAs) can help if you’re not getting enough nutrients from food."
    },
    {
      "question": "What are the best exercises for building core strength?",
      "answer": "Effective core exercises include planks, Russian twists, hanging leg raises, and deadbugs. Compound movements like squats and deadlifts also strengthen the core indirectly."
    },
    {
      "question": "How can I prevent injuries during weightlifting?",
      "answer": "Prevent injuries by warming up properly, using correct form, avoiding overloading, and gradually increasing weight. Also, incorporate mobility exercises and ensure adequate rest between sessions."
    },
    {
      "question": "What is the difference between strength training and hypertrophy training?",
      "answer": "Strength training focuses on increasing the amount of weight you can lift, typically with low reps (1-5). Hypertrophy training focuses on building muscle size, usually with moderate weight and higher reps (6-12)."
    },
    {
      "question": "How can I improve my pull-up performance?",
      "answer": "To improve pull-ups, work on grip strength, lat activation, and body control. Start with assisted pull-ups, negatives, or lat pull-downs, and progressively reduce assistance as you get stronger."
    },
    {
      "question": "What’s the difference between free weights and machines?",
      "answer": "Free weights engage stabilizing muscles and allow for a full range of motion, making them more effective for functional strength. Machines are easier to control and may be better for beginners or isolation work."
    },
    {
      "question": "What are the signs of overtraining?",
      "answer": "Signs of overtraining include fatigue, poor performance, decreased motivation, frequent injuries, difficulty sleeping, and prolonged muscle soreness. Rest and recovery are key to avoiding burnout."
    },
    {
      "question": "How many sets and reps should I do to build strength?",
      "answer": "For strength, aim for 3-5 sets of 3-6 reps with heavy weights. Focus on compound lifts like squats, deadlifts, and bench presses, and take longer rest periods (2-5 minutes) between sets."
    },
    {
      "question": "What’s the best way to warm up before lifting weights?",
      "answer": "Start with 5-10 minutes of light cardio to raise your heart rate, followed by dynamic stretches or mobility exercises targeting the muscles you’ll be working. Include lighter sets of your main lifts to practice form."
    },
    {
      "question": "How can I increase my bench press strength?",
      "answer": "To increase bench press strength, focus on improving technique, increase pressing volume, strengthen supporting muscles like triceps and shoulders, and implement progressive overload."
    },
    {
      "question": "How long should I rest between sets?",
      "answer": "Rest periods depend on your goal. For strength, rest 2-5 minutes between sets. For hypertrophy (muscle growth), rest 60-90 seconds. For endurance, keep rests short at 30-60 seconds."
    },
    {
      "question": "What are some good recovery tips after a tough workout?",
      "answer": "Post-workout recovery should include hydration, protein-rich meals, stretching, and adequate sleep. Consider foam rolling to relieve muscle tightness and promote circulation."
    },
    {
      "question": "What is progressive overload?",
      "answer": "Progressive overload is the gradual increase of stress placed on muscles during training. This can be achieved by increasing the weight lifted, reps performed, or the intensity of the workout."
    },
    {
      "question": "What are the benefits of compound exercises?",
      "answer": "Compound exercises, like squats and deadlifts, engage multiple muscle groups simultaneously. They improve functional strength, increase calorie burn, and are efficient for both strength and muscle growth."
    },
    {
      "question": "How should I structure my weekly workout routine?",
      "answer": "A balanced workout routine could include 3-4 strength training sessions, focusing on different muscle groups each day, and 1-2 cardio or HIIT sessions. Incorporate flexibility and mobility work as well."
    },
    {
      "question": "Is it okay to work out every day?",
      "answer": "It's okay to work out daily as long as you vary the intensity and focus on different muscle groups to avoid overtraining. Incorporate active recovery days with lighter activities like walking or yoga."
    },
    {
      "question": "What’s the best time of day to work out?",
      "answer": "The best time to work out depends on personal preference and schedule. Some people prefer mornings for energy boosts, while others perform better in the afternoon or evening due to body temperature and muscle readiness."
    },
    {
      "question": "How important is hydration during exercise?",
      "answer": "Hydration is crucial during exercise to maintain performance and prevent dehydration. Drink water before, during, and after your workout, especially if you’re exercising in hot conditions or sweating heavily."
    },
    {
      "question": "What’s the best way to lose belly fat?",
      "answer": "Spot reduction isn't possible, so to lose belly fat, focus on overall fat loss through a combination of strength training, cardio, and a calorie deficit. Nutrition and consistency are key factors."
    },
    {
      "question": "How do I know if I’m lifting the right weight?",
      "answer": "The right weight should challenge you but allow you to maintain proper form. If you can perform all reps without struggling, increase the weight. If your form breaks down, lower the weight."
    },
    {
      "question": "What are the benefits of stretching after a workout?",
      "answer": "Stretching after a workout helps improve flexibility, reduces muscle stiffness, and promotes relaxation. It also aids in muscle recovery by improving blood circulation."
    },
    {
      "question": "What is the difference between bulking and cutting?",
      "answer": "Bulking is a phase where you consume more calories than you burn to gain muscle mass. Cutting is the opposite, where you reduce calorie intake to lose fat while trying to preserve muscle."
    },
    {
      "question": "How can I improve my running endurance?",
      "answer": "To improve running endurance, gradually increase your running distance or time, incorporate interval training, and make sure to rest and recover between runs. Stay consistent and build up slowly."
    },
    {
      "question": "How important is sleep for muscle growth?",
      "answer": "Sleep is critical for muscle growth and recovery. During sleep, your body releases growth hormones and repairs muscle tissue. Aim for 7-9 hours of quality sleep each night for optimal recovery."
    },
    {
      "question": "Should I do cardio before or after weightlifting?",
      "answer": "It depends on your goal. If strength or muscle growth is your priority, do cardio after weightlifting to avoid fatigue. If cardio is your main focus, you can do it first."
    },
    {
      "question": "What are some effective exercises for improving balance?",
      "answer": "Effective balance exercises include single-leg squats, standing on one leg with eyes closed, and using unstable surfaces like a Bosu ball or wobble board to challenge your stability."
    },
    {
      "question": "How do I track my fitness progress?",
      "answer": "Track your progress by logging your workouts, taking body measurements, and monitoring strength gains. Photos and tracking body fat percentage can also provide insight into your progress."
    },
    {
      "question": "What’s the difference between fasted and fed cardio?",
      "answer": "Fasted cardio is done on an empty stomach, usually in the morning, with the goal of burning more fat. Fed cardio is performed after eating, which may provide more energy for a better performance."
    },
    {
      "question": "How can I improve my flexibility?",
      "answer": "To improve flexibility, incorporate static stretching after your workouts, and try yoga or Pilates. Focus on stretching the major muscle groups at least 3-4 times a week for optimal results."
    },
    {
      "question": "What are the benefits of using resistance bands?",
      "answer": "Resistance bands are great for improving strength, flexibility, and muscle endurance. They’re portable, versatile, and can be used to target multiple muscle groups with low impact on joints."
    },
    {
      "question": "How do I reduce fat while maintaining muscle?",
      "answer": "To lose fat while preserving muscle, eat a high-protein diet, engage in strength training, and maintain a moderate calorie deficit. Prioritize compound lifts and avoid excessive cardio, which may lead to muscle loss."
    },
    {
      "question": "What are some good exercises for targeting the glutes?",
      "answer": "Effective glute exercises include squats, lunges, hip thrusts, deadlifts, and glute bridges. Focus on full range of motion and controlled movements for optimal glute activation."
    },
    {
      "question": "How can I stay motivated to work out consistently?",
      "answer": "To stay motivated, set specific and achievable goals, track your progress, find a workout routine you enjoy, and consider working out with a friend or hiring a trainer for accountability."
    },
    {
      "question": "What are some good pre-workout snacks?",
      "answer": "Good pre-workout snacks include a banana with peanut butter, Greek yogurt with fruit, or a protein bar. Choose foods that are easy to digest and provide both carbs and protein for energy."
    },
    {
      "question": "How can I improve my grip strength?",
      "answer": "Improve grip strength by incorporating exercises like deadlifts, farmer's carries, pull-ups, and using grip trainers. Over time, grip strength will increase, improving overall performance in weightlifting."
    },
    {
      "question": "How often should I change my workout routine?",
      "answer": "To avoid plateaus, change your workout routine every 6-8 weeks. This can involve altering exercises, rep ranges, or intensity levels to continue challenging your muscles."
    },
    
    {
        "question": "What are the benefits of doing deadlifts?",
        "answer": "Deadlifts strengthen the posterior chain, including your lower back, hamstrings, glutes, and traps. They improve overall strength, posture, and grip while also increasing muscle mass and functional fitness."
      },
      {
        "question": "How often should I do cardio to lose fat?",
        "answer": "For fat loss, aim for 3-5 sessions of cardio per week, each lasting 30-45 minutes. Combine steady-state cardio with high-intensity interval training (HIIT) for the best results."
      },
      {
        "question": "What's the best way to build muscle?",
        "answer": "To build muscle, focus on progressive overload with compound exercises like squats, deadlifts, bench presses, and rows. Train each muscle group 2-3 times per week, prioritize protein intake, and ensure adequate rest and recovery."
      },
      {
        "question": "What should I eat before a workout?",
        "answer": "Eat a balanced meal 1-3 hours before your workout that includes complex carbs, protein, and some fats. For example, oatmeal with a scoop of protein powder or a chicken sandwich with whole grain bread."
      },
      {
        "question": "How much protein should I consume daily?",
        "answer": "For muscle building, consume about 1.6-2.2 grams of protein per kilogram of body weight per day. For general health and maintenance, aim for 1.0-1.2 grams per kilogram."
      },
      {
        "question": "What's the proper form for a bench press?",
        "answer": "Keep your feet flat on the floor, arch your lower back slightly, and keep your shoulder blades retracted. Lower the bar to your mid-chest, keep your elbows at a 45-degree angle, and push the bar up in a controlled motion."
      },
      {
        "question": "Can I lose fat and build muscle at the same time?",
        "answer": "Yes, but it requires a careful balance of nutrition and training. Focus on strength training, maintain a slight calorie deficit, and ensure sufficient protein intake to support muscle growth while losing fat."
      },
      {
        "question": "What are the benefits of HIIT?",
        "answer": "HIIT improves cardiovascular fitness, burns fat efficiently, and preserves muscle mass. It's time-efficient and boosts metabolism for hours after the workout due to the EPOC (Excess Post-exercise Oxygen Consumption) effect."
      },
      {
        "question": "How can I improve my squat depth?",
        "answer": "To improve squat depth, work on ankle mobility, hip flexibility, and core stability. Incorporate exercises like goblet squats, hip flexor stretches, and ankle dorsiflexion drills."
      },
      {
        "question": "How do I prevent muscle soreness after a workout?",
        "answer": "To reduce muscle soreness, stay hydrated, perform a proper warm-up and cool-down, and include stretching. Active recovery, such as light walking or swimming, can also help alleviate soreness."
      },
      {
        "question": "Is it necessary to take supplements for muscle growth?",
        "answer": "Supplements are not mandatory for muscle growth, but they can support your diet. Protein powder, creatine, and branched-chain amino acids (BCAAs) can help if you're not getting enough nutrients from food."
      },
      {
        "question": "What are the best exercises for building core strength?",
        "answer": "Effective core exercises include planks, Russian twists, hanging leg raises, and deadbugs. Compound movements like squats and deadlifts also strengthen the core indirectly."
      },
      {
        "question": "How can I prevent injuries during weightlifting?",
        "answer": "Prevent injuries by warming up properly, using correct form, avoiding overloading, and gradually increasing weight. Also, incorporate mobility exercises and ensure adequate rest between sessions."
      },
      {
        "question": "What is the difference between strength training and hypertrophy training?",
        "answer": "Strength training focuses on increasing the amount of weight you can lift, typically with low reps (1-5). Hypertrophy training focuses on building muscle size, usually with moderate weight and higher reps (6-12)."
      },
      {
        "question": "How can I improve my pull-up performance?",
        "answer": "To improve pull-ups, work on grip strength, lat activation, and body control. Start with assisted pull-ups, negatives, or lat pull-downs, and progressively reduce assistance as you get stronger."
      },
      {
        "question": "What's the difference between free weights and machines?",
        "answer": "Free weights engage stabilizing muscles and allow for a full range of motion, making them more effective for functional strength. Machines are easier to control and may be better for beginners or isolation work."
      },
      {
        "question": "What are the signs of overtraining?",
        "answer": "Signs of overtraining include fatigue, poor performance, decreased motivation, frequent injuries, difficulty sleeping, and prolonged muscle soreness. Rest and recovery are key to avoiding burnout."
      },
      {
        "question": "How many sets and reps should I do to build strength?",
        "answer": "For strength, aim for 3-5 sets of 3-6 reps with heavy weights. Focus on compound lifts like squats, deadlifts, and bench presses, and take longer rest periods (2-5 minutes) between sets."
      },
      {
        "question": "What's the best way to warm up before lifting weights?",
        "answer": "Start with 5-10 minutes of light cardio to raise your heart rate, followed by dynamic stretches or mobility exercises targeting the muscles you'll be working. Include lighter sets of your main lifts to practice form."
      },
      {
        "question": "How can I increase my bench press strength?",
        "answer": "To increase bench press strength, focus on improving technique, increase pressing volume, strengthen supporting muscles like triceps and shoulders, and implement progressive overload."
      },
      {
        "question": "How long should I rest between sets?",
        "answer": "Rest periods depend on your goal. For strength, rest 2-5 minutes between sets. For hypertrophy (muscle growth), rest 60-90 seconds. For endurance, keep rests short at 30-60 seconds."
      },
      {
        "question": "What are some good recovery tips after a tough workout?",
        "answer": "Post-workout recovery should include hydration, protein-rich meals, stretching, and adequate sleep. Consider foam rolling to relieve muscle tightness and promote circulation."
      },
      {
        "question": "What is progressive overload?",
        "answer": "Progressive overload is the gradual increase of stress placed on muscles during training. This can be achieved by increasing the weight lifted, reps performed, or the intensity of the workout."
      },
      {
        "question": "What are the benefits of compound exercises?",
        "answer": "Compound exercises, like squats and deadlifts, engage multiple muscle groups simultaneously. They improve functional strength, increase calorie burn, and are efficient for both strength and muscle growth."
      },
      {
        "question": "How should I structure my weekly workout routine?",
        "answer": "A balanced workout routine could include 3-4 strength training sessions, focusing on different muscle groups each day, and 1-2 cardio or HIIT sessions. Incorporate flexibility and mobility work as well."
      },
      {
        "question": "Is it okay to work out every day?",
        "answer": "It's okay to work out daily as long as you vary the intensity and focus on different muscle groups to avoid overtraining. Incorporate active recovery days with lighter activities like walking or yoga."
      },
      {
        "question": "What's the best time of day to work out?",
        "answer": "The best time to work out depends on personal preference and schedule. Some people prefer mornings for energy boosts, while others perform better in the afternoon or evening due to body temperature and muscle readiness."
      },
      {
        "question": "How important is hydration during exercise?",
        "answer": "Hydration is crucial during exercise to maintain performance and prevent dehydration. Drink water before, during, and after your workout, especially if you're exercising in hot conditions or sweating heavily."
      },
      {
        "question": "What's the best way to lose belly fat?",
        "answer": "Spot reduction isn't possible, so to lose belly fat, focus on overall fat loss through a combination of strength training, cardio, and a calorie deficit. Nutrition and consistency are key factors."
      },
      {
        "question": "How do I know if I'm lifting the right weight?",
        "answer": "The right weight should challenge you but allow you to maintain proper form. If you can perform all reps without struggling, increase the weight. If your form breaks down, lower the weight."
      },
      {
        "question": "What are the benefits of stretching after a workout?",
        "answer": "Stretching after a workout helps improve flexibility, reduces muscle stiffness, and promotes relaxation. It also aids in muscle recovery by improving blood circulation."
      },
      {
        "question": "What is the difference between bulking and cutting?",
        "answer": "Bulking is a phase where you consume more calories than you burn to gain muscle mass. Cutting is the opposite, where you reduce calorie intake to lose fat while trying to preserve muscle."
      },
      {
        "question": "How can I improve my running endurance?",
        "answer": "To improve running endurance, gradually increase your running distance or time, incorporate interval training, and make sure to rest and recover between runs. Stay consistent and build up slowly."
      },
      {
        "question": "How important is sleep for muscle growth?",
        "answer": "Sleep is critical for muscle growth and recovery. During sleep, your body releases growth hormones and repairs muscle tissue. Aim for 7-9 hours of quality sleep each night for optimal recovery."
      },
      {
        "question": "Should I do cardio before or after weightlifting?",
        "answer": "It depends on your goal. If strength or muscle growth is your priority, do cardio after weightlifting to avoid fatigue. If cardio is your main focus, you can do it first."
      },
      {
        "question": "What are some effective exercises for improving balance?",
        "answer": "Effective balance exercises include single-leg squats, standing on one leg with eyes closed, and using unstable surfaces like a Bosu ball or wobble board to challenge your stability."
      },
      {
        "question": "How do I track my fitness progress?",
        "answer": "Track your progress by logging your workouts, taking body measurements, and monitoring strength gains. Photos and tracking body fat percentage can also provide insight into your progress."
      },
      {
        "question": "What's the difference between fasted and fed cardio?",
        "answer": "Fasted cardio is done on an empty stomach, usually in the morning, with the goal of burning more fat. Fed cardio is performed after eating, which may provide more energy for a better performance."
      },
      {
        "question": "How can I improve my flexibility?",
        "answer": "To improve flexibility, incorporate static stretching after your workouts, and try yoga or Pilates. Focus on stretching the major muscle groups at least 3-4 times a week for optimal results."
      },
      {
        "question": "What are the benefits of using resistance bands?",
        "answer": "Resistance bands are great for improving strength, flexibility, and muscle endurance. They're portable, versatile, and can be used to target multiple muscle groups with low impact on joints."
      },
      {
        "question": "How do I reduce fat while maintaining muscle?",
        "answer": "To lose fat while preserving muscle, eat a high-protein diet, engage in strength training, and maintain a moderate calorie deficit. Prioritize compound lifts and avoid excessive cardio, which may lead to muscle loss."
      },
      {
        "question": "What are some good exercises for targeting the glutes?",
        "answer": "Effective glute exercises include squats, lunges, hip thrusts, deadlifts, and glute bridges. Focus on full range of motion and controlled movements for optimal glute activation."
      },
      {
        "question": "How can I stay motivated to work out consistently?",
        "answer": "To stay motivated, set specific and achievable goals, track your progress, find a workout routine you enjoy, and consider working out with a friend or hiring a trainer for accountability."
      },
      {
        "question": "What are some good pre-workout snacks?",
        "answer": "Good pre-workout snacks include a banana with peanut butter, Greek yogurt with fruit, or a protein bar. Choose foods that are easy to digest and provide both carbs and protein for energy."
      },
      {
        "question": "How can I improve my grip strength?",
        "answer": "Improve grip strength by incorporating exercises like deadlifts, farmer's carries, pull-ups, and using grip trainers. Over time, grip strength will increase, improving overall performance in weightlifting."
      },
      {
        "question": "How often should I change my workout routine?",
        "answer": "To avoid plateaus, change your workout routine every 6-8 weeks. This can involve altering exercises, rep ranges, or intensity levels to continue challenging your muscles."
      },
      {
        "question": "What's the proper technique for Olympic lifts like the clean and jerk?",
        "answer": "Olympic lifts require proper form and progression. Start with learning the power clean, focus on explosive hip drive, proper foot positioning, and maintaining a straight back. Work with a qualified coach initially to master the technique."
      },
      {
        "question": "How can I improve my vertical jump?",
        "answer": "To improve vertical jump, focus on plyometric exercises like box jumps and depth jumps, strengthen your legs with squats and deadlifts, and work on explosive movements. Include core work and proper landing mechanics training."
      },
      {
        "question": "What are the benefits of training with kettlebells?",
        "answer": "Kettlebells provide full-body workouts, improve functional strength, and enhance mobility. They're excellent for developing power, core stability, and cardiovascular endurance through exercises like swings and Turkish get-ups."
      },
      {
        "question": "How do I properly track my macronutrients?",
        "answer": "Use a food tracking app to log meals, weigh portions, and monitor protein, carbs, and fats. Aim for consistency rather than perfection, and adjust based on your goals and body's response."
      },
      {
        "question": "What's the importance of tempo in weight training?",
        "answer": "Tempo training controls the speed of repetitions, affecting muscle tension and growth. Slower negatives increase time under tension, while explosive positives build power. Vary tempos for different training effects."
      },
      {
        "question": "What are the benefits of supersets?",
        "answer": "Supersets increase workout efficiency, enhance muscle endurance, boost metabolic rate, and can help break through plateaus. They're particularly effective for hypertrophy training and time-restricted workouts."
      },
      {
        "question": "How do I properly perform a front squat?",
        "answer": "Maintain an upright torso, keep elbows high, and position the bar across your deltoids. Focus on core engagement and proper breathing throughout the movement. Start with lighter weights to master form."
      },
      {
        "question": "What's the role of vitamin D in fitness?",
        "answer": "Vitamin D supports bone health, muscle function, and hormone production. Adequate levels can improve strength, reduce injury risk, and enhance recovery. Consider supplementation if exposure to sunlight is limited."
      },
      {
        "question": "How can I improve my shoulder mobility?",
        "answer": "Practice shoulder dislocations with a resistance band, wall slides, and rotator cuff exercises. Include thoracic spine mobility work and maintain good posture throughout the day."
      },
      {
        "question": "What are the benefits of cross-training?",
        "answer": "Cross-training prevents boredom, reduces injury risk, improves overall fitness, and helps break through plateaus. It also develops different energy systems and movement patterns."
      },
      {
        "question": "What's the proper way to perform barbell rows?",
        "answer": "Hinge at hips, keep back straight, pull bar to lower chest while keeping elbows close to body. Focus on squeezing shoulder blades together at the top and maintain core stability throughout."
      },
      {
        "question": "How do I properly measure body fat percentage?",
        "answer": "Common methods include calipers, bioelectrical impedance, DEXA scans, and hydrostatic weighing. For consistency, use the same method and measure at the same time of day under similar conditions."
      },
      {
        "question": "What are the benefits of cluster sets?",
        "answer": "Cluster sets involve brief rest periods between reps, allowing heavier weights with better form. They improve power output, technique, and total volume while reducing fatigue during heavy lifting."
      },
      {
        "question": "How do I improve my ankle mobility?",
        "answer": "Practice ankle dorsiflexion stretches, calf raises with slow eccentrics, and banded ankle mobilizations. Use a foam roller on calves and perform weight-bearing stretches regularly."
      },
      {
        "question": "What's the importance of post-workout nutrition timing?",
        "answer": "Consume protein and carbs within 30-60 minutes post-workout to optimize recovery and muscle protein synthesis. This window isn't as critical if you're eating regular meals throughout the day."
      },
      {
        "question": "How can I develop better mind-muscle connection?",
        "answer": "Start with lighter weights, focus on feeling the target muscle work, use mirrors for form check, and incorporate pause reps. Practice isolation exercises before compound movements."
      },
      {
        "question": "What are the benefits of Olympic weightlifting?",
        "answer": "Olympic lifting develops explosive power, improves coordination, increases strength and mobility, and enhances athletic performance. It also burns significant calories and builds functional muscle."
      },
      {
        "question": "How do I properly perform face pulls?",
        "answer": "Set cable at head height, pull toward face while externally rotating shoulders, focus on rear deltoids and upper back engagement. Keep elbows high and controlled throughout movement."
      },
      {
        "question": "What's the importance of deload weeks?",
        "answer": "Deload weeks reduce physical and mental fatigue, allow for recovery and adaptation, and prevent plateaus. Typically reduce volume or intensity by 40-60% for one week every 4-8 weeks."
      },
      {
        "question": "How can I improve my hip mobility?",
        "answer": "Practice hip flexor stretches, 90/90 mobility work, pigeon pose, and dynamic hip stretches. Include exercises like deep squats and lunges while maintaining proper form."
      },
      {
        "question": "What are the benefits of isometric exercises?",
        "answer": "Isometrics build strength at specific joint angles, improve mind-muscle connection, and are joint-friendly. They're effective for rehabilitation and can be done without equipment."
      },
      {
        "question": "How do I properly perform dips?",
        "answer": "Keep shoulders packed, slight forward lean for chest focus, maintain controlled descent, and avoid excessive flaring of elbows. Progress from assisted to full body weight gradually."
      },
      {
        "question": "What's the role of magnesium in fitness?",
        "answer": "Magnesium supports muscle function, energy production, and recovery. It helps reduce muscle cramps, improve sleep quality, and regulate nerve function. Consider supplementation if deficient."
      },
      {
        "question": "How can I improve my sprinting speed?",
        "answer": "Focus on proper running mechanics, strengthen hip extensors and flexors, practice acceleration drills, and include plyometric exercises. Work on both stride length and frequency."
      },
      {
        "question": "What are the benefits of band pull-aparts?",
        "answer": "Band pull-aparts strengthen rear deltoids, improve posture, and enhance shoulder stability. They're excellent for warming up and can be done frequently throughout the day."
      },
      {
        "question": "How do I properly perform landmine presses?",
        "answer": "Start in athletic stance, maintain core stability, press weight in an arc motion while rotating torso. Great for shoulder development and core engagement with reduced joint stress."
      },
      {
        "question": "What's the importance of proper breathing during lifts?",
        "answer": "Proper breathing stabilizes core, increases intra-abdominal pressure, and improves power output. Generally exhale during exertion and inhale during the eccentric phase."
      },
      {
        "question": "How can I improve my muscle definition?",
        "answer": "Focus on reducing body fat through caloric deficit while maintaining muscle mass with resistance training. Ensure adequate protein intake and include both compound and isolation exercises."
      },
      {
        "question": "What are the benefits of battle ropes?",
        "answer": "Battle ropes provide high-intensity cardiovascular training, build upper body endurance, and improve core stability. They're excellent for conditioning and can be used for various movement patterns."
      },
      {
        "question": "How do I properly perform Romanian deadlifts?",
        "answer": "Maintain slight knee bend, hinge at hips while keeping back straight, lower weight while feeling hamstring stretch. Focus on hip movement and maintain bar close to legs."
      },
      {
        "question": "What's the role of BCAAs in training?",
        "answer": "BCAAs (Branched-Chain Amino Acids) can help reduce muscle soreness, prevent muscle breakdown during intense training, and support recovery. Most beneficial when training fasted."
      },
      {
        "question": "How can I improve my posture while lifting?",
        "answer": "Strengthen core and back muscles, practice proper bracing techniques, and maintain neutral spine alignment. Focus on scapular retraction and proper head position during exercises."
      },
      {
        "question": "What are the benefits of medicine ball training?",
        "answer": "Medicine balls develop explosive power, improve rotational strength, enhance core stability, and are great for functional fitness. They're versatile tools for both strength and conditioning."
      },
      {
        "question": "How do I properly perform skull crushers?",
        "answer": "Keep upper arms stable, lower weight to forehead level, focus on triceps engagement. Maintain control throughout movement and avoid elbow flare."
      },
      {
        "question": "What's the importance of joint mobility?",
        "answer": "Joint mobility improves movement quality, reduces injury risk, enhances performance, and supports better exercise form. Regular mobility work should be part of every training program."
      },
      {
        "question": "How do I maximize muscle growth during a bulk?",
        "answer": "Focus on progressive overload, maintain a caloric surplus of 300-500 calories, consume 2g protein per kg bodyweight, prioritize compound movements, and ensure 7-9 hours of quality sleep."
      },
      {
        "question": "What's the proper way to do a muscle-up?",
        "answer": "Start with explosive pull-up, transition smoothly over bar with false grip, finish with dip. Master strict pull-ups and straight bar dips first, then practice transition phase separately."
      },
      {
        "question": "How can I improve my overhead mobility?",
        "answer": "Work on thoracic extension, practice wall slides, incorporate shoulder dislocates, and stretch lats regularly. Address any limitations in shoulder blade movement and rotator cuff strength."
      },
      {
        "question": "What are the benefits of circus dumbbell training?",
        "answer": "Circus dumbbells develop unilateral strength, improve stability and control, enhance grip strength, and build explosive power. They're excellent for strongman training and functional strength."
      },
      {
        "question": "How do I properly perform a zercher squat?",
        "answer": "Cradle bar in elbow crease, maintain upright torso, keep core tight throughout movement. Focus on proper breathing and maintain elbow position to prevent bar slipping."
      },
      {
        "question": "What's the role of carb cycling in fitness?",
        "answer": "Carb cycling involves alternating high and low carb days to optimize body composition, maintain training intensity, and manage insulin sensitivity. Align higher carb days with intense training sessions."
      },
      {
        "question": "How can I improve my power clean technique?",
        "answer": "Start with proper setup, focus on explosive triple extension, keep bar close to body, and catch in front rack position. Practice position drills and technique work before adding heavy weight."
      },
      {
        "question": "What are the benefits of loaded carries?",
        "answer": "Loaded carries build total body strength, improve core stability, enhance grip strength, and develop functional fitness. They're excellent for real-world strength and posture improvement."
      },
      {
        "question": "How do I properly perform ring dips?",
        "answer": "Stabilize rings at lockout, maintain false grip if possible, control descent, and keep elbows in. Progress from supported to full ring dips gradually while maintaining shoulder control."
      },
      {
        "question": "What's the importance of eccentric training?",
        "answer": "Eccentric training increases muscle damage and growth stimulus, improves tendon strength, and enhances overall strength. It's particularly effective for breaking through plateaus."
      },
      {
        "question": "How can I improve my handstand pushup?",
        "answer": "Build shoulder strength with pike pushups, practice wall handstands, work on shoulder mobility and stability. Progress from negative reps to full range of motion gradually."
      },
      {
        "question": "What are the benefits of sand training?",
        "answer": "Sand training increases energy expenditure, improves stabilizer muscle strength, reduces impact stress, and enhances proprioception. It's excellent for conditioning and injury prevention."
      },
      {
        "question": "How do I properly perform heavy bag training?",
        "answer": "Maintain proper form with punches, incorporate footwork, vary combinations, and include both power and speed work. Focus on technique before increasing intensity."
      },
      {
        "question": "What's the role of electrolytes in training?",
        "answer": "Electrolytes maintain proper hydration, support muscle function, prevent cramping, and aid in recovery. Particularly important during long or intense training sessions in hot conditions."
      },
      {
        "question": "How can I improve my muscle symmetry?",
        "answer": "Include unilateral exercises, address mobility restrictions, perform regular assessments, and focus on mind-muscle connection. Use mirrors and video feedback to check form."
      },
      {
        "question": "What are the benefits of pause reps?",
        "answer": "Pause reps eliminate momentum, improve technique, increase time under tension, and enhance mind-muscle connection. They're particularly effective for breaking through sticking points."
      },
      {
        "question": "How do I properly perform box squats?",
        "answer": "Set appropriate box height, control descent, pause briefly without relaxing, drive through heels to stand. Maintain tension throughout movement and focus on explosive concentric phase."
      },
      {
        "question": "What's the importance of workout periodization?",
        "answer": "Periodization organizes training into specific phases, optimizes progression, prevents plateaus, and reduces injury risk. It allows for planned variation in volume, intensity, and focus."
      },
      {
        "question": "How can I improve my front lever?",
        "answer": "Build pulling strength, develop core control, practice progression holds like tuck and advanced tuck. Include specific scapular strengthening and straight-arm pulling exercises."
      },
      {
        "question": "What are the benefits of complexes?",
        "answer": "Complexes combine multiple exercises performed consecutively, improve conditioning, burn calories efficiently, and enhance movement patterns. They're excellent for time-efficient workouts."
      },
      {
        "question": "How do I properly perform deficit deadlifts?",
        "answer": "Stand on elevated platform, maintain proper starting position, focus on controlled eccentric. These increase range of motion and help develop strength off the floor."
      },
      {
        "question": "What's the role of omega-3s in fitness?",
        "answer": "Omega-3s support recovery, reduce inflammation, improve joint health, and enhance nutrient absorption. Consider supplementation if fish intake is low."
      },
      {
        "question": "How can I improve my explosive strength?",
        "answer": "Incorporate plyometrics, Olympic lifts, medicine ball throws, and band-resisted movements. Focus on quality of movement and full triple extension."
      },
      {
        "question": "What are the benefits of suspension training?",
        "answer": "Suspension trainers develop core stability, improve balance, allow for varied angles of resistance, and enhance functional strength. They're versatile tools for bodyweight training."
      },
      {
        "question": "How do I properly perform good mornings?",
        "answer": "Maintain slight knee bend, hinge at hips while keeping back straight, focus on hamstring stretch. Keep bar properly positioned across upper back and control the movement."
      },
      {
        "question": "What's the importance of intra-workout nutrition?",
        "answer": "Intra-workout nutrition can help maintain performance during long sessions, prevent muscle breakdown, and support hydration. Most beneficial during training sessions exceeding 90 minutes."
      },
      {
        "question": "How can I improve my muscle activation?",
        "answer": "Use proper warm-up sets, practice isolation exercises, incorporate paused reps, and focus on mind-muscle connection. Consider using EMG biofeedback when available."
      },
      {
        "question": "What are the benefits of contrast showers?",
        "answer": "Alternating hot and cold water can improve circulation, reduce muscle soreness, enhance recovery, and boost immune function. Start with shorter durations and build tolerance gradually."
      },
      {
        "question": "What's the proper way to perform barbell hip thrusts?",
        "answer": "Position upper back on bench, roll barbell over hips with padding, plant feet firmly, and drive hips up by squeezing glutes. Maintain neutral spine and avoid overarching lower back."
      },
      {
        "question": "How do I incorporate drop sets effectively?",
        "answer": "Start with your working weight, perform to near failure, quickly reduce weight by 20-30%, and continue. Usually best used at the end of a workout for 1-2 exercises to avoid overtraining."
      },
      {
        "question": "What's the role of creatine loading?",
        "answer": "Creatine loading involves taking 20g daily for 5-7 days to saturate muscles quickly. While effective, you can achieve similar results with 5g daily over a longer period without loading."
      },
      {
        "question": "How can I improve my overhead squat?",
        "answer": "Work on shoulder mobility, ankle flexibility, and thoracic spine extension. Practice with PVC pipe before adding weight, focus on maintaining active shoulders throughout movement."
      },
      {
        "question": "What are the benefits of blood flow restriction training?",
        "answer": "BFR training can stimulate muscle growth with lighter weights, reduce joint stress, and enhance recovery. Best used as a supplement to traditional training, not a replacement."
      },
      {
        "question": "How do I properly perform reverse lunges?",
        "answer": "Step back while maintaining upright posture, lower back knee toward ground, keep front knee stable. Focus on control and balance, drive through front heel to return to start."
      },
      {
        "question": "What's the importance of training grip styles?",
        "answer": "Different grip styles (double overhand, mixed, hook) target forearms differently and affect maximum lift capacity. Regularly training various grips improves overall strength and prevents plateaus."
      },
      {
        "question": "How can I improve my muscle recovery rate?",
        "answer": "Optimize sleep quality, maintain proper nutrition, stay hydrated, use active recovery techniques, and consider recovery tools like compression garments or massage."
      },
      {
        "question": "What are the benefits of anti-rotation exercises?",
        "answer": "Anti-rotation exercises strengthen core stability, improve sports performance, prevent lower back pain, and enhance movement control. Examples include Pallof presses and renegade rows."
      },
      {
        "question": "How do I properly perform face pulls?",
        "answer": "Set cable at head height, use rope attachment, pull toward face while externally rotating shoulders. Focus on squeezing rear delts and maintaining elbow position above wrists."
      },
      {
        "question": "What's the role of tempo training?",
        "answer": "Tempo training controls movement speed to target different muscle fibers and training adaptations. Common format is eccentric-pause-concentric-pause (e.g., 4-1-2-0)."
      },
      {
        "question": "How can I improve my muscle ups?",
        "answer": "Build pulling strength with weighted pull-ups, work on explosive power, practice transition phase separately. Master false grip and straight bar dips before combining movements."
      },
      {
        "question": "What are the benefits of isometric holds?",
        "answer": "Isometric holds improve strength at specific joint angles, enhance mind-muscle connection, and are joint-friendly. Effective for breaking through plateaus and rehabilitation."
      },
      {
        "question": "How do I properly perform Bulgarian split squats?",
        "answer": "Place rear foot on elevated surface, maintain upright posture, lower until front thigh is parallel to ground. Keep front knee tracked over toe and control the movement."
      },
      {
        "question": "What's the importance of scapular control?",
        "answer": "Proper scapular control prevents shoulder injuries, improves lifting performance, and enhances upper body strength. Essential for pull-ups, pushups, and overhead movements."
      },
      {
        "question": "How can I improve my Turkish get-up?",
        "answer": "Break down the movement into components, practice each phase separately, maintain eye contact with weight. Focus on shoulder stability and core control throughout."
      },
      {
        "question": "What are the benefits of training unilaterally?",
        "answer": "Unilateral training identifies and corrects imbalances, improves stabilizer strength, enhances coordination, and can help prevent injuries. Also increases core engagement."
      },
      {
        "question": "How do I properly perform pendlay rows?",
        "answer": "Start with bar on ground, maintain horizontal back position, pull explosively to lower chest. Reset completely between reps and focus on controlled eccentric phase."
      },
      {
        "question": "What's the role of post-activation potentiation?",
        "answer": "PAP involves using heavy loads briefly to enhance performance in subsequent explosive movements. Example: heavy squats followed by jump squats for improved power output."
      },
      {
        "question": "How can I improve my front lever?",
        "answer": "Progress through tuck, advanced tuck, single leg, and straddle positions. Build straight-arm pulling strength and core control with specific accessory exercises."
      }
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
