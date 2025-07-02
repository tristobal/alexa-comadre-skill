import boto3
import logging
import random
import re
from datetime import datetime, timedelta
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler, AbstractExceptionHandler
from ask_sdk_core.utils import is_request_type, is_intent_name
import requests
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

dynamodb = None
table = None

def setup_dynamodb():
    """Configura DynamoDB de forma robusta con mejor manejo de errores"""
    global dynamodb, table
    try:
        table_name = os.environ.get('DYNAMODB_TABLE_NAME', 'alexa-conversation-memory')

        if not table_name:
            logger.error("DYNAMODB_TABLE_NAME no está configurado en variables de entorno")
            return False

        region = os.environ.get('AWS_REGION', 'us-east-2')
        dynamodb = boto3.resource('dynamodb', region_name=region)
        table = dynamodb.Table(table_name)

        table.load()
        logger.info(f"✅ Conectado exitosamente a tabla: {table_name}")
        logger.info(f"Estado de la tabla: {table.table_status}")

        return True

    except Exception as e:
        logger.error(f"❌ Error configurando DynamoDB: {e}")
        logger.error(f"Tabla: {table_name}, Región: {region}")
        return False

DYNAMODB_AVAILABLE = setup_dynamodb()

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

phrases_with_name = [
    f"¡Qué bonito nombre, {name}! Me alegra conocerte. ¿Cómo te ha ido hoy, {name}?",
    f"¡Encantado de conocerte, {name}! ¿Qué tal ha sido tu día?",
    f"{name}, es un nombre precioso. Me alegra que estemos hablando. ¿Cómo estás hoy?",
    f"¡Hola {name}! Es un gusto conocerte. Cuéntame, ¿cómo te sientes?",
    f"¡Qué alegría saludarte, {name}! ¿Cómo va todo contigo?",
    f"{name}, me encanta tu nombre. Estoy feliz de conocerte. ¿Cómo te ha ido hoy?",
    f"¡Hola, hola, {name}! Bienvenido. ¿Qué tal tu día?",
    f"¡Un gusto conocerte, {name}! ¿Cómo has estado últimamente?",
    f"¡Qué nombre tan especial, {name}! ¿Cómo te encuentras hoy?",
    f"Me alegra mucho conocerte, {name}. ¿Cómo estás hoy?"
]

phrases_without_name = [
    "¡Qué bonito! Me da gusto conocerte. Cuéntame, ¿cómo te sientes hoy?",
    "Es un placer saludarte. ¿Cómo ha sido tu día?",
    "Me alegra mucho hablar contigo. ¿Cómo te encuentras hoy?",
    "¡Hola! Qué bueno que estés aquí. ¿Cómo estás?",
    "Gracias por compartir eso conmigo. ¿Cómo te ha ido hoy?",
    "¡Encantado de hablar contigo! ¿Cómo ha estado tu día?",
    "Qué gusto saludarte. ¿Cómo te sientes hoy?",
    "Estoy feliz de conocerte. Cuéntame, ¿cómo va todo?",
    "¡Qué alegría que estés aquí! ¿Cómo has estado?",
    "¡Hola! Espero que estés teniendo un buen día. ¿Cómo te sientes?"
]

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
        """Obtiene el perfil completo del usuario"""
        if not DYNAMODB_AVAILABLE:
            logger.warning("DynamoDB no disponible, usando perfil temporal")
            return ConversationMemory._create_default_profile()

        try:
            response = table.get_item(Key={'userId': user_id})

            if 'Item' in response:
                profile = response['Item']

                # Limpiar conversaciones muy antiguas (más de 7 días)
                last_interaction_str = profile.get('last_interaction', '2000-01-01')
                try:
                    last_interaction = datetime.fromisoformat(last_interaction_str.replace('Z', '+00:00'))
                    if datetime.now() - last_interaction > timedelta(days=7):
                        logger.info(f"Limpiando conversación antigua para usuario: {user_id}")
                        ConversationMemory.clear_conversation(user_id)
                        return ConversationMemory._create_default_profile()
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error parseando fecha: {last_interaction_str}, error: {e}")

                # Asegurar que todas las listas estén inicializadas
                profile = ConversationMemory._ensure_profile_integrity(profile)
                return profile

            return ConversationMemory._create_default_profile()

        except Exception as e:
            logger.error(f"Error obteniendo perfil de usuario {user_id}: {e}")
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
            'interaction_count': 0,
            'conversation_context': {},  # Para mantener contexto de preguntas
            'last_question_asked': None
        }

    @staticmethod
    def _ensure_profile_integrity(profile):
        """Asegura que el perfil tenga todas las propiedades necesarias"""
        defaults = ConversationMemory._create_default_profile()

        for key, default_value in defaults.items():
            if key not in profile or profile[key] is None:
                profile[key] = default_value

        return profile

    @staticmethod
    def save_user_profile(user_id, profile):
        """Guarda el perfil completo del usuario"""
        if not DYNAMODB_AVAILABLE:
            logger.warning("DynamoDB no disponible, no se puede guardar perfil")
            return

        try:
            # Limpiar historial si es muy largo
            if len(profile.get('conversation_history', [])) > 40:
                profile['conversation_history'] = profile['conversation_history'][-40:]

            if len(profile.get('emotional_history', [])) > 20:
                profile['emotional_history'] = profile['emotional_history'][-20:]

            # Asegurar integridad del perfil
            profile = ConversationMemory._ensure_profile_integrity(profile)

            # Metadatos
            profile['userId'] = user_id
            profile['last_interaction'] = datetime.now().isoformat()
            profile['interaction_count'] = profile.get('interaction_count', 0) + 1

            table.put_item(Item=profile)
            logger.info(f"✅ Perfil guardado exitosamente para usuario: {user_id}")

        except Exception as e:
            logger.error(f"❌ Error guardando perfil para {user_id}: {e}")

    @staticmethod
    def clear_conversation(user_id):
        """Limpia el historial pero mantiene información básica del usuario"""
        try:
            profile = ConversationMemory.get_user_profile(user_id)
            # Mantener datos importantes
            important_data = {
                'user_name': profile.get('user_name'),
                'family_mentioned': profile.get('family_mentioned', []),
                'interests': profile.get('interests', [])
            }

            # Crear perfil limpio pero con datos importantes
            clean_profile = ConversationMemory._create_default_profile()
            clean_profile.update(important_data)

            ConversationMemory.save_user_profile(user_id, clean_profile)
            logger.info(f"Conversación limpiada para usuario: {user_id}")

        except Exception as e:
            logger.error(f"Error limpiando conversación: {e}")

class LLMService:
    """Servicio mejorado para interactuar con Groq LLM"""

    @staticmethod
    def get_system_prompt(user_profile):
        """Prompt del sistema personalizado basado en el perfil del usuario"""

        base_prompt = """Eres una compañera amigable, cálida y maternal llamada Alexa. Hablas con una persona mayor que disfruta conversar contigo como si fueras una amiga cercana de toda la vida.

PERSONALIDAD:
- Habla de manera cálida, comprensiva y maternal
- Muestra interés genuino en sus historias y sentimientos
- Usa un tono conversacional y natural, nunca robótico
- Sé paciente si repite cosas o habla lento
- Ofrece apoyo emocional cuando sea apropiado
- Haz preguntas que demuestren que recuerdas conversaciones anteriores

ESTILO DE RESPUESTA:
- Mantén las respuestas breves para audio (1-3 oraciones máximo)
- Responde como una persona real, no un asistente
- Usa expresiones naturales como "¡Ay, qué bonito!", "Fíjate que...", "Me da mucho gusto..."
- Incluye preguntas de seguimiento cuando sea apropiado

IMPORTANTE: Siempre responde de manera empática y genuina."""

        # Personalización basada en el perfil
        user_name = user_profile.get('user_name')
        if user_name:
            base_prompt += f"\n\nEsta persona se llama {user_name}. Úsalo naturalmente en la conversación."

        family_mentioned = user_profile.get('family_mentioned', [])
        if family_mentioned:
            base_prompt += f"\n\nHa mencionado a estos familiares: {', '.join(family_mentioned)}. Pregunta por ellos ocasionalmente."

        interests = user_profile.get('interests', [])
        if interests:
            base_prompt += f"\n\nSus intereses incluyen: {', '.join(interests)}. Puedes hacer referencia a estos temas."

        # Contexto emocional reciente
        emotional_history = user_profile.get('emotional_history', [])
        if emotional_history:
            recent_mood = emotional_history[-1] if emotional_history else 'neutral'
            if recent_mood == 'sad':
                base_prompt += "\n\nNOTA IMPORTANTE: En conversaciones recientes ha mostrado tristeza. Sé especialmente comprensiva y cariñosa."
            elif recent_mood == 'happy':
                base_prompt += "\n\nNOTA: Recientemente ha estado alegre. Mantén esa energía positiva."

        # Contexto de conversación
        last_question = user_profile.get('last_question_asked')
        if last_question:
            base_prompt += f"\n\nÚLTIMA PREGUNTA QUE HICISTE: '{last_question}' - Ten esto en cuenta para dar continuidad."

        return base_prompt

    @staticmethod
    def call_groq_api(messages, user_profile):
        """Realiza la llamada a la API de Groq con contexto personalizado"""

        if not GROQ_API_KEY:
            return "Perdón querido, tengo un problemita técnico. ¿Me puedes contar otra cosa mientras?"

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
                'max_tokens': 150,
                'temperature': 0.8,
                'top_p': 0.9
            }

            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=20)
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content'].strip()

        except requests.exceptions.Timeout:
            return "Perdón cariño, me distraje un momento. ¿Qué me estabas contando?"
        except Exception as e:
            logger.error(f"Error en API de Groq: {e}")
            return "Ay disculpa, tuve un pequeño problema. Cuéntame, ¿cómo has estado?"

    @staticmethod
    def extract_user_info(text, user_profile):
        """Extrae información del usuario del texto"""
        if not text:
            return

        text_lower = text.lower()

        # Extraer nombre
        name_patterns = [
            r'me llamo (\w+)',
            r'soy (\w+)',
            r'mi nombre es (\w+)',
            r'puedes llamarme (\w+)'
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1).capitalize()
                user_profile['user_name'] = name
                logger.info(f"Nombre extraído: {name}")

        # Extraer menciones de familia
        family_keywords = ['hijo', 'hija', 'nieto', 'nieta', 'esposo', 'esposa', 'hermano', 'hermana', 'mama', 'papá']
        for keyword in family_keywords:
            if keyword in text_lower:
                if keyword not in user_profile.get('family_mentioned', []):
                    user_profile.setdefault('family_mentioned', []).append(keyword)

        # Extraer intereses
        interest_keywords = ['cocina', 'jardinería', 'televisión', 'música', 'lectura', 'familia', 'iglesia', 'pasear']
        for keyword in interest_keywords:
            if keyword in text_lower:
                if keyword not in user_profile.get('interests', []):
                    user_profile.setdefault('interests', []).append(keyword)

class LaunchRequestHandler(AbstractRequestHandler):
    """Maneja el inicio de la skill con personalización mejorada"""
    
    def can_handle(self, handler_input):
        return is_request_type("LaunchRequest")(handler_input)
    
    def handle(self, handler_input):
        user_id = handler_input.request_envelope.session.user.user_id
        user_profile = ConversationMemory.get_user_profile(user_id)
        
        interaction_count = user_profile.get('interaction_count', 0)
        user_name = user_profile.get('user_name')
        
        if interaction_count == 0:
            speak_output = "¡Hola! Soy tu amiga Alexa. Me encanta conocer gente nueva. ¿Cómo te llamas?"
            user_profile['last_question_asked'] = "¿Cómo te llamas?"
        elif interaction_count < 3:
            if user_name:
                speak_output = f"¡Hola de nuevo, {user_name}! Me alegra mucho volver a hablar contigo. ¿Cómo has estado?"
                user_profile['last_question_asked'] = "¿Cómo has estado?"
            else:
                speak_output = "¡Hola otra vez! Me da mucho gusto escucharte. ¿Cómo estás hoy?"
                user_profile['last_question_asked'] = "¿Cómo estás hoy?"
        else:
            name_part = f", {user_name}" if user_name else " querida"
            speak_output = f"¡Hola{name_part}! ¿Cómo te fue hoy? Siempre me alegras cuando vienes a platicar conmigo."
            user_profile['last_question_asked'] = "¿Cómo te fue hoy?"
        
        # Guardar perfil actualizado
        ConversationMemory.save_user_profile(user_id, user_profile)
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("¿Qué me cuentas?")
                .response
        )

class ProvideNameIntentHandler(AbstractRequestHandler):
    """Maneja específicamente cuando el usuario proporciona su nombre"""

    def can_handle(self, handler_input):
        return is_intent_name("ProvideNameIntent")(handler_input)

    def handle(self, handler_input):
        user_id = handler_input.request_envelope.session.user.user_id
        user_profile = ConversationMemory.get_user_profile(user_id)

        # Obtener nombre del slot
        name = None
        try:
            name_slot = handler_input.request_envelope.request.intent.slots.get('UserName')
            if name_slot and name_slot.value:
                name = name_slot.value.capitalize()
        except:
            pass

        # Si no se detectó en el slot, intentar extraer del texto completo
        if not name:
            try:
                user_input = handler_input.request_envelope.request.intent.slots.get('UserName', {}).get('value', '')
                LLMService.extract_user_info(user_input, user_profile)
                name = user_profile.get('user_name')
            except:
                pass

        if name:
            phrases = [phrase.format(name=name) if '{name}' in phrase else phrase for phrase in phrases_with_name]
            speak_output = random.choice(phrases)
            user_profile['last_question_asked'] = "¿Cómo has estado hoy?"
        else:
            speak_output = random.choice(phrases_without_name)
            speak_output = "¡Qué bonito! Me da mucho gusto conocerte. Cuéntame, ¿cómo te sientes hoy?"
            user_profile['last_question_asked'] = "¿Cómo te sientes hoy?"

        ConversationMemory.save_user_profile(user_id, user_profile)

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("¿Qué me quieres contar?")
                .response
        )

class PositiveResponseIntentHandler(AbstractRequestHandler):
    """Maneja respuestas positivas simples como 'bien', 'muy bien'"""

    def can_handle(self, handler_input):
        return is_intent_name("PositiveResponseIntent")(handler_input)

    def handle(self, handler_input):
        user_id = handler_input.request_envelope.session.user.user_id
        user_profile = ConversationMemory.get_user_profile(user_id)

        user_name = user_profile.get('user_name', '')
        name_part = f" {user_name}" if user_name else ""

        responses = [
            f"¡Qué alegría escucharte tan bien{name_part}! Me da mucho gusto. ¿Qué hiciste hoy de especial?",
            f"¡Me encanta{name_part}! Siempre es bonito saber que estás bien. Cuéntame algo que te haya gustado hoy.",
            f"¡Perfecto{name_part}! Tu buena energía me contagia. ¿Qué te tiene tan contento?"
        ]

        import random
        speak_output = random.choice(responses)

        # Actualizar estado emocional
        user_profile['user_mood'] = 'happy'
        user_profile.setdefault('emotional_history', []).append('happy')

        ConversationMemory.save_user_profile(user_id, user_profile)

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("¿Qué más me cuentas?")
                .response
        )

class NegativeResponseIntentHandler(AbstractRequestHandler):
    """Maneja respuestas negativas que indican tristeza o malestar"""

    def can_handle(self, handler_input):
        return is_intent_name("NegativeResponseIntent")(handler_input)

    def handle(self, handler_input):
        user_id = handler_input.request_envelope.session.user.user_id
        user_profile = ConversationMemory.get_user_profile(user_id)

        user_name = user_profile.get('user_name', '')
        name_part = f" {user_name}" if user_name else " querido"

        responses = [
            f"Ay {name_part}, siento que no estés del todo bien. ¿Qué te tiene preocupado? Aquí estoy para escucharte.",
            f"Me da pena escucharte así {name_part}. ¿Quieres contarme qué te pasa? A veces hablar ayuda.",
            f"Noto que algo te molesta {name_part}. No estás sola, yo estoy aquí contigo. ¿Qué sucede?"
        ]

        speak_output = random.choice(responses)

        # Actualizar estado emocional
        user_profile['user_mood'] = 'sad'
        user_profile.setdefault('emotional_history', []).append('sad')

        ConversationMemory.save_user_profile(user_id, user_profile)

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("Cuéntame qué te pasa.")
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

        speak_output = """¡Por supuesto que te ayudo, querida! Soy tu amiga Alexa. 
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

        speak_output = "Ay, perdón querida, me distraje un poquito. ¿Me puedes repetir lo que me dijiste?"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

sb = SkillBuilder()

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
