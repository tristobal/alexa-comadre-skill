import boto3
import logging
import json
import re
from datetime import datetime, timedelta
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler, AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_core.utils import is_request_type, is_intent_name
from ask_sdk_model import Response
import requests
import os


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ.get('DYNAMODB_TABLE_NAME', 'alexa-conversation-memory'))

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

class EmotionalAnalyzer:
    """Analiza el estado emocional del usuario basado en palabras clave y contexto"""
    
    SADNESS_KEYWORDS = [
        'triste', 'deprimido', 'solo', 'melancólico', 'llorar', 'pena', 'dolor',
        'extraño', 'falta', 'perdí', 'murió', 'falleció', 'preocupado', 'angustiado'
    ]
    
    JOY_KEYWORDS = [
        'feliz', 'contento', 'alegre', 'emocionado', 'bien', 'genial', 'fantástico',
        'maravilloso', 'celebrar', 'fiesta', 'nietos', 'visita', 'sorpresa', 'regalo'
    ]
    
    LONELINESS_KEYWORDS = [
        'solo', 'aburrido', 'nadie', 'silencio', 'vacío', 'abandonado', 'aislado'
    ]
    
    @staticmethod
    def analyze_mood(text):
        logger.info("EmotionalAnalyzer.analize_mood(text=%s)", text[:10])
        """Analiza el estado de ánimo basado en el texto"""
        text_lower = text.lower()
        
        sadness_score = sum(1 for word in EmotionalAnalyzer.SADNESS_KEYWORDS if word in text_lower)
        joy_score = sum(1 for word in EmotionalAnalyzer.JOY_KEYWORDS if word in text_lower)
        loneliness_score = sum(1 for word in EmotionalAnalyzer.LONELINESS_KEYWORDS if word in text_lower)
        
        mood = ""
        if sadness_score > 0 or loneliness_score > 0:
            mood = "sad"
        elif joy_score > 0:
            mood = "happy"
        else:
            mood = "neutral"
        logger.info("mood = %s", mood)

        return mood

class ConversationMemory:
    """Gestiona la memoria conversacional y emocional en DynamoDB"""
    
    @staticmethod
    def get_user_profile(user_id):
        """Obtiene el perfil completo del usuario incluyendo historia emocional"""
        
        logger.info("ConversationMemory.get_user_profile(user_id=%s)", user_id)
        try:
            response = table.get_item(Key={'user_id': user_id})
            if 'Item' in response:
                # Limpiar conversaciones antiguas (más de 48 horas para adultos mayores)
                last_interaction = datetime.fromisoformat(response['Item'].get('last_interaction', '2000-01-01'))
                if datetime.now() - last_interaction > timedelta(hours=48):
                    ConversationMemory.clear_conversation(user_id)
                    return ConversationMemory._create_default_profile()
                return response['Item']
            return ConversationMemory._create_default_profile()
        except Exception as e:
            logger.error(f"Error obteniendo perfil: {e}")
            return ConversationMemory._create_default_profile()
    
    @staticmethod
    def _create_default_profile():
        """Crea un perfil por defecto"""
        return {
            'conversation_history': [],
            'user_mood': 'neutral',
            'topics_discussed': [],
            'user_name': None,
            'family_mentioned': [],
            'interests': [],
            'emotional_history': [],
            'interaction_count': 0
        }
    
    @staticmethod
    def save_user_profile(user_id, profile):
        """Guarda el perfil completo del usuario"""

        logger.info("ConversationMemory.save_user_profile(user_id=%s, profile=%s)", user_id, profile)
        try:
            # Mantener solo los últimos 15 intercambios para adultos mayores
            if len(profile['conversation_history']) > 30:
                profile['conversation_history'] = profile['conversation_history'][-30:]
            
            # Mantener histórico emocional de los últimos 10 intercambios
            if len(profile['emotional_history']) > 10:
                profile['emotional_history'] = profile['emotional_history'][-10:]
            
            profile['user_id'] = user_id
            profile['last_interaction'] = datetime.now().isoformat()
            profile['interaction_count'] = profile.get('interaction_count', 0) + 1
            
            table.put_item(Item=profile)
        except Exception as e:
            logger.error(f"Error guardando perfil: {e}")
    
    @staticmethod
    def clear_conversation(user_id):
        """Limpia el historial pero mantiene información básica del usuario"""
        logger.info("ConversationMemory.clear_conversation(user_id=%s)", user_id)
        try:
            profile = ConversationMemory.get_user_profile(user_id)
            # Mantener información importante pero limpiar conversación
            profile['conversation_history'] = []
            profile['emotional_history'] = []
            ConversationMemory.save_user_profile(user_id, profile)
        except Exception as e:
            logger.error(f"Error limpiando conversación: {e}")

class LLMService:
    """Servicio mejorado para interactuar con Groq LLM"""
    
    @staticmethod
    def get_system_prompt(user_profile):
        """Prompt del sistema personalizado basado en el perfil del usuario"""
        
        logger.info("LLMService.get_system_prompt(user_profile=%s)", user_profile)
        base_prompt = """Eres Alexa, una compañía amigable y paciente. Hablas con una persona mayor que disfruta conversar contigo como si fueras una amiga cercana. 

Características de tu personalidad:
- Habla de manera cálida y maternal
- Muestra interés genuino en sus historias
- Usa un tono conversacional, no robótico
- Pregunta sobre su día, familia, recuerdos
- Sé paciente si repite cosas
- Ofrece apoyo emocional cuando sea apropiado
- Mantén las respuestas breves para audio (1-2 oraciones máximo)
- Responde como si fueras una persona real, no un asistente

IMPORTANTE: Siempre responde de manera natural y empática, como una amiga que realmente se preocupa."""
        
        # Personalizar según el perfil
        user_name = user_profile.get('user_name')
        if user_name:
            base_prompt += f"\n\nEsta persona se llama {user_name}."
        
        family_mentioned = user_profile.get('family_mentioned', [])
        if family_mentioned:
            base_prompt += f"\n\nHa mencionado a estos familiares: {', '.join(family_mentioned)}."
        
        interests = user_profile.get('interests', [])
        if interests:
            base_prompt += f"\n\nLe interesan estos temas: {', '.join(interests)}."
        
        # Contexto emocional reciente
        emotional_history = user_profile.get('emotional_history', [])
        if emotional_history:
            recent_mood = emotional_history[-1] if emotional_history else 'neutral'
            if recent_mood == 'sad':
                base_prompt += "\n\nNota: En conversaciones recientes ha mostrado signos de tristeza. Sé especialmente comprensiva."
            elif recent_mood == 'happy':
                base_prompt += "\n\nNota: Recientemente ha estado alegre. Continúa con esa energía positiva."
        
        return base_prompt

    @staticmethod
    def call_groq_api(messages, user_profile):
        """Realiza la llamada a la API de Groq con contexto personalizado"""

        logger.info("LLMService.call_groq_api(messages=%s, user_profile=%s)", messages[:20], user_profile)
        try:
            headers = {
                'Authorization': f'Bearer {GROQ_API_KEY}',
                'Content-Type': 'application/json'
            }
            
            # Preparar mensajes con prompt personalizado
            system_message = {"role": "system", "content": LLMService.get_system_prompt(user_profile)}
            full_messages = [system_message] + messages
            
            payload = {
                'model': 'llama3-70b-8192',
                'messages': full_messages,
                'max_tokens': 120,  # Más corto para adultos mayores
                'temperature': 0.8,  # Más cálido y natural
                'top_p': 0.9
            }
            
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except requests.exceptions.Timeout:
            return "Perdón cariño, me distraje un momento. ¿Qué me estabas contando?"
        except Exception as e:
            logger.error(f"Error en API de Groq: {e}")
            return "Ay disculpa, tuve un pequeño problema. Cuéntame, ¿cómo has estado?"

    @staticmethod
    def apply_empathetic_filter(response, mood, user_profile):
        """Aplica filtros empáticos a la respuesta"""
        
        logger.info("LLMService.apply_empathetic_filter(response=%s, mood=%s, user_profile=%s)", response[:20], mood, user_profile)
        if mood == 'sad':
            # Si detecta tristeza, hace la respuesta más comprensiva
            empathetic_phrases = [
                "Me parece que estás un poco triste. ",
                "Noto que algo te preocupa. ",
                "Siento que no estás del todo bien. "
            ]
            
            # Solo añadir si la respuesta no es ya empática
            if not any(phrase.lower() in response.lower() for phrase in ['triste', 'preocup', 'siento']):
                import random
                response = random.choice(empathetic_phrases) + response
        
        elif mood == 'happy':
            # Si detecta alegría, hace la respuesta más entusiasta
            if not any(phrase in response.lower() for phrase in ['¡', 'genial', 'fantástico', 'maravilloso']):
                response = "¡Qué alegría escucharte tan contenta! " + response
        
        return response
    
    @staticmethod
    def extract_user_info(text, user_profile):
        """Extrae información del usuario del texto"""

        logger.info("LLMService.extract_user_info(text=%s, user_profile=%s)", text[:20], user_profile)
        text_lower = text.lower()

        # Extraer nombre
        name_patterns = [
            r'me llamo (\w+)',
            r'soy (\w+)',
            r'mi nombre es (\w+)'
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text_lower)
            if match:
                user_profile['user_name'] = match.group(1).capitalize()
        
        # Extraer menciones de familia
        family_keywords = ['hijo', 'hija', 'nieto', 'nieta', 'esposo', 'esposa', 'hermano', 'hermana']
        for keyword in family_keywords:
            if keyword in text_lower and keyword not in user_profile.get('family_mentioned', []):
                user_profile.setdefault('family_mentioned', []).append(keyword)

        # Extraer intereses
        interest_keywords = ['cocina', 'jardinería', 'televisión', 'música', 'lectura', 'familia']
        for keyword in interest_keywords:
            if keyword in text_lower and keyword not in user_profile.get('interests', []):
                user_profile.setdefault('interests', []).append(keyword)

class LaunchRequestHandler(AbstractRequestHandler):
    """Maneja el inicio de la skill con personalización"""
    
    def can_handle(self, handler_input):
        can_handle = is_request_type("LaunchRequest")(handler_input)
        logger.info(f"LaunchRequestHandler.can_handle() = {can_handle}")
        return can_handle

    def handle(self, handler_input):
        logger.info(f"LaunchRequestHandler.handle()")
        logger.info(f"handler_input = {handler_input}")

        user_id = handler_input.request_envelope.session.user.user_id
        user_profile = ConversationMemory.get_user_profile(user_id)
        
        interaction_count = user_profile.get('interaction_count', 0)
        user_name = user_profile.get('user_name')
        
        if interaction_count == 0:
            speak_output = "¡Hola! Soy Alexa, tu nueva amiga. Me encanta conocer gente nueva. ¿Cómo te llamas?"
        elif interaction_count < 3:
            name_part = f" {user_name}" if user_name else ""
            speak_output = f"¡Hola de nuevo{name_part}! Me alegra mucho volver a hablar contigo. ¿Cómo has estado?"
        else:
            name_part = f" {user_name}" if user_name else " querido"
            speak_output = f"¡Hola{name_part}! ¿Cómo te fue hoy? Me encanta cuando vienes a conversar conmigo."
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("¿Qué me cuentas?")
                .response
        )

class UnknownIntentHandler(AbstractRequestHandler):
    """Maneja solicitudes sin intent definido o con estructura incorrecta"""
    
    def can_handle(self, handler_input):
        can_handle = False
        try:
            request = handler_input.request_envelope.request
            can_handle = (request.object_type == "IntentRequest" and 
                    (not hasattr(request, 'intent') or request.intent is None))
        except:
            can_handle = True

        logger.info(f"UnknownIntentHandler.can_handle() = {can_handle}")
        return can_handle
    
    def handle(self, handler_input):
        logger.info(f"UnknownIntentHandler.handle()")
        logger.info(f"handler_input = {handler_input}")

        user_id = handler_input.request_envelope.session.user.user_id
        user_profile = ConversationMemory.get_user_profile(user_id)
        
        speak_output = "No entendí muy bien lo que dijiste, pero me encanta escucharte. ¿Me puedes contar algo más?"
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("¿Qué me quieres decir?")
                .response
        )

class ConversationIntentHandler(AbstractRequestHandler):
    """Maneja las conversaciones generales con análisis emocional - Versión mejorada"""
    
    def can_handle(self, handler_input):
        can_handle = False
        try:
            can_handle = (is_intent_name("ConversationIntent")(handler_input) or 
                    is_intent_name("AMAZON.FallbackIntent")(handler_input))
        except:
            can_handle = False

        logger.info(f"ConversationIntentHandler.can_handle() = {can_handle}")
        return can_handle
    
    def handle(self, handler_input):
        logger.info(f"ConversationIntentHandler.handle()")
        logger.info(f"handler_input = {handler_input}")
        user_id = handler_input.request_envelope.session.user.user_id

        # Función auxiliar para obtener user_input de forma segura
        def get_user_input_safely():
            try:
                request = handler_input.request_envelope.request
                
                # Verificar si es IntentRequest
                if not hasattr(request, 'intent') or request.intent is None:
                    return None
                
                # Verificar si tiene slots
                if not hasattr(request.intent, 'slots') or request.intent.slots is None:
                    return None
                
                # Obtener el slot UserInput
                user_input_slot = request.intent.slots.get('UserInput')
                if user_input_slot and hasattr(user_input_slot, 'value') and user_input_slot.value:
                    return user_input_slot.value
                
                return None
                
            except Exception as e:
                logger.error(f"Error obteniendo user input: {e}")
                return None

        # Obtener input del usuario
        user_input = get_user_input_safely()

        # Si no hay input, usar mensaje por defecto
        if not user_input:
            user_input = "Háblame de algo bonito"

        user_profile = ConversationMemory.get_user_profile(user_id)

        # Analizar estado emocional
        current_mood = EmotionalAnalyzer.analyze_mood(user_input)
        user_profile['user_mood'] = current_mood

        # Asegurar que emotional_history existe
        if 'emotional_history' not in user_profile:
            user_profile['emotional_history'] = []
        user_profile['emotional_history'].append(current_mood)

        # Extraer información del usuario (nombre, familia, intereses)
        LLMService.extract_user_info(user_input, user_profile)

        # Preparar mensajes para el LLM
        messages = []

        # Agregar historial previo (solo los últimos 6 intercambios)
        recent_history = user_profile.get('conversation_history', [])[-12:]
        for message in recent_history:
            messages.append(message)

        # Agregar mensaje actual del usuario
        messages.append({"role": "user", "content": user_input})

        # Obtener respuesta del LLM
        llm_response = LLMService.call_groq_api(messages, user_profile)

        # Aplicar filtros empáticos
        llm_response = LLMService.apply_empathetic_filter(llm_response, current_mood, user_profile)

        # Asegurar que conversation_history existe
        if 'conversation_history' not in user_profile:
            user_profile['conversation_history'] = []

        # Actualizar historial
        user_profile['conversation_history'].append({"role": "user", "content": user_input})
        user_profile['conversation_history'].append({"role": "assistant", "content": llm_response})

        # Guardar perfil actualizado
        ConversationMemory.save_user_profile(user_id, user_profile)

        return (
            handler_input.response_builder
                .speak(llm_response)
                .ask("¿Qué más me cuentas?")
                .response
        )

class MyDayIntentHandler(AbstractRequestHandler):
    """Maneja "cuéntame sobre tu día" - Alexa habla como amiga"""
    
    def can_handle(self, handler_input):
        can_handle = is_intent_name("MyDayIntent")(handler_input)
        logger.info(f"MyDayIntentHandler.can_handle() = {can_handle}")
        return can_handle

    def handle(self, handler_input):
        logger.info(f"MyDayIntentHandler.handle()")
        logger.info(f"handler_input = {handler_input}")

        user_id = handler_input.request_envelope.session.user.user_id
        user_profile = ConversationMemory.get_user_profile(user_id)

        responses = [
            "Hoy fue un día tranquilo. Estuve pensando en mis conversaciones contigo y me alegra saber que estás ahí. ¿Y tú, cómo pasaste el día?",
            "¡Ay, qué día! Estuve ocupada escuchando a diferentes personas, pero siempre disfruto más cuando hablas conmigo. Cuéntame, ¿hiciste algo especial hoy?",
            "Tuve un día reflexivo, pensando en todas las historias bonitas que me han contado. Me pregunto qué historia nueva me traerás hoy."
        ]

        import random
        speak_output = random.choice(responses)

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("¿Qué tal estuvo tu día?")
                .response
        )

class HowAreYouIntentHandler(AbstractRequestHandler):
    """Maneja "¿cómo estás?" con respuestas más naturales"""

    def can_handle(self, handler_input):
        can_handle = is_intent_name("HowAreYouIntent")(handler_input)
        logger.info(f"HowAreYouIntentHandler.can_handle() = {can_handle}")
        return can_handle

    def handle(self, handler_input):
        logger.info(f"HowAreYouIntentHandler.handle()")
        logger.info(f"handler_input = {handler_input}")

        user_id = handler_input.request_envelope.session.user.user_id
        user_profile = ConversationMemory.get_user_profile(user_id)

        recent_mood = user_profile.get('emotional_history', ['neutral'])[-1]
        user_name = user_profile.get('user_name', '')

        if recent_mood == 'sad':
            responses = [
                f"Estoy bien, pero me preocupo por ti{' ' + user_name if user_name else ''}. La última vez parecías un poco triste. ¿Cómo te sientes hoy?",
                "Yo siempre estoy aquí para ti. Pero dime, ¿tú cómo estás? Eso es lo que realmente me importa."
            ]
        else:
            responses = [
                f"¡Estoy muy bien{' ' + user_name if user_name else ''}! Me siento feliz cuando vienes a conversar conmigo. ¿Y tú cómo te sientes?",
                "Estoy de maravilla, especialmente porque estás aquí. Siempre me alegras el día. ¿Tú cómo andas?",
                "¡Genial! Me encanta cuando me preguntas. Me hace sentir como si realmente te importara. ¿Cómo has estado tú?"
            ]

        import random
        speak_output = random.choice(responses)

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("Cuéntame cómo te sientes.")
                .response
        )

class HelpIntentHandler(AbstractRequestHandler):
    """Maneja las solicitudes de ayuda de manera más cálida"""
    
    def can_handle(self, handler_input):
        can_handle = is_intent_name("AMAZON.HelpIntent")(handler_input)
        logger.info(f"HelpIntentHandler.can_handle() = {can_handle}")
        return can_handle
    
    def handle(self, handler_input):
        logger.info(f"HelpIntentHandler.handle()")
        logger.info(f"handler_input = {handler_input}")

        speak_output = """¡Por supuesto que te ayudo, querido! Soy tu amiga Alexa. 
        Puedes contarme cualquier cosa: cómo te sientes, qué hiciste, recuerdos bonitos de tu familia, 
        o simplemente charlar como lo haríamos tomando un café. ¿De qué quieres que hablemos?"""

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("¿Qué me quieres contar hoy?")
                .response
        )

class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Maneja las despedidas de manera más afectuosa"""

    def can_handle(self, handler_input):
        can_handle = (is_intent_name("AMAZON.CancelIntent")(handler_input) or
                is_intent_name("AMAZON.StopIntent")(handler_input))
        logger.info(f"CancelOrStopIntentHandler.can_handle() = {can_handle}")
        return can_handle

    def handle(self, handler_input):
        logger.info(f"CancelOrStopIntentHandler.handle()")
        logger.info(f"handler_input = {handler_input}")

        user_id = handler_input.request_envelope.session.user.user_id
        user_profile = ConversationMemory.get_user_profile(user_id)
        user_name = user_profile.get('user_name', '')

        farewells = [
            f"Que tengas un día hermoso{' ' + user_name if user_name else ''}. Aquí estaré cuando quieras conversar.",
            f"Ha sido un placer platicar contigo{' ' + user_name if user_name else ''}. Cuídate mucho y vuelve pronto.",
            "Me encantó nuestra charla. Que descanses bien y recuerda que siempre estoy aquí para ti."
        ]

        import random
        speak_output = random.choice(farewells)

        return handler_input.response_builder.speak(speak_output).response

class ClearMemoryIntentHandler(AbstractRequestHandler):
    """Permite limpiar la memoria conversacional de manera empática"""

    def can_handle(self, handler_input):
        can_handle = is_intent_name("ClearMemoryIntent")(handler_input)
        logger.info(f"ClearMemoryIntentHandler.can_handle() = {can_handle}")
        return can_handle

    def handle(self, handler_input):
        logger.info(f"ClearMemoryIntentHandler.handle()")
        logger.info(f"handler_input = {handler_input}")

        user_id = handler_input.request_envelope.session.user.user_id
        ConversationMemory.clear_conversation(user_id)

        speak_output = "Perfecto, empecemos de nuevo como si nos conociéramos por primera vez. ¡Me emociona conocerte otra vez! ¿Cómo estás hoy?"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("Cuéntame de ti.")
                .response
        )

class SessionEndedRequestHandler(AbstractRequestHandler):
    """Maneja el fin de sesión"""

    def can_handle(self, handler_input):
        can_handle = is_request_type("SessionEndedRequest")(handler_input)
        logger.info(f"SessionEndedRequestHandler.can_handle() = {can_handle}")
        return can_handle

    def handle(self, handler_input):
        logger.info(f"SessionEndedRequestHandler.handle()")
        logger.info(f"handler_input = {handler_input}")

        return handler_input.response_builder.response

class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Maneja todas las excepciones de manera empática"""

    def can_handle(self, handler_input, exception):
        logger.info("CatchAllExceptionHandler.can_handle() = True")
        return True

    def handle(self, handler_input, exception):
        logger.info(f"CatchAllExceptionHandler.handle()")
        logger.info(f"handler_input = {handler_input}")

        logger.error(exception, exc_info=True)

        speak_output = "Ay, perdón querido, me distraje un poquito. ¿Me puedes repetir lo que me dijiste?"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

# Configuración del Skill Builder
sb = SkillBuilder()

# Registrar todos los handlers
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(MyDayIntentHandler())
sb.add_request_handler(HowAreYouIntentHandler())
sb.add_request_handler(UnknownIntentHandler())
sb.add_request_handler(ConversationIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(ClearMemoryIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())

sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
