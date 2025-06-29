# Alexa Skill: Mi Comadre - Empathic Companion

Alexa Skill designed especially as an empathic companion for older adults, using LLM (Groq API) with emotional analysis and personalized memory.

## 🌟 Key Features.

### Empathic Personality.
- **Maternal and warm tone**: Speaks like a close friend, not a robotic assistant.
- Emotional memory**: Remembers the user's mood from previous conversations.
- Adaptive Responses**: Adjusts its tone according to the detected emotional context

### Specific Functionalities

#### 1. Natural Conversation
- **Invocation**: "Alexa, open my comadre!"
- **Examples**: 
  - "I want to tell you something."
  - "I'm feeling sad today"
  - "I have to tell you that..."

#### 2. Alexa as Friend.
- **"Alexa, tell me about your day "** → Alexa responds as if she were a real person.
- **"Alexa, how are you? "** → Natural and reciprocal responses.

#### 3. Intelligent Emotional Analysis.
- **Sadness detection**: "You seem a little sad to me today, do you want to talk about it?"
- **Happiness detection**: "How nice to hear you so happy! Tell me more."
- **Loneliness detection**: Especially sympathetic responses.

### Personalized Memory

The skill maintains a complete user profile that includes:

```python
user_profile = {
 "user_name": "Maria", # User name
 "conversation_history": [...], # Conversation history
 "user_mood": "happy",
```

### Requirements
- AWS Lambda previously configured
- IAM credentials with permissions to update Lambda

> Please check this link for deploy.yaml https://docs.aws.amazon.com/lambda/latest/dg/functions-states.html