import asyncio
import boto3
import logging
import random
import re
import os
from datetime import datetime

from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler, AbstractExceptionHandler
from ask_sdk_core.utils import is_request_type, is_intent_name
from googletrans import Translator
from textblob import TextBlob

import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- CONFIGURACIÓN DE SERVICIOS ---

dynamodb = None
table = None
translator = Translator()

def setup_services():
    """Inicializa DynamoDB y el analizador de sentimientos una sola vez."""
    global dynamodb, table, sentiment_analyzer
    
    # Configurar DynamoDB
    try:
        if not table:
            table_name = os.environ.get('DYNAMODB_TABLE_NAME', 'alexa-virtual-companion')
            region = os.environ.get('AWS_REGION', 'us-east-1')
            dynamodb = boto3.resource('dynamodb', region_name=region)
            table = dynamodb.Table(table_name)
            table.load()
            logger.info(f"✅ Conectado a la tabla de DynamoDB: {table_name}")
    except Exception as e:
        logger.error(f"❌ Error CRÍTICO al configurar DynamoDB: {e}")
        # Si DynamoDB falla, la skill no puede funcionar correctamente.
        # En un escenario real, podrías manejar esto de forma más elegante.


setup_services()

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class EmotionalAnalyzer:
    """Clase para analizar el estado emocional usando un modelo de ML."""

    @staticmethod
    async def _translate_to_eng(text):
        translation = await translator.translate(text, src='es', dest='en')
        return translation.text

    @staticmethod
    def analyze_mood(text):
        try:
            eng_txt = EmotionalAnalyzer._translate_to_eng(text)
            blob = TextBlob(eng_txt)
            polarity = blob.sentiment.polarity
            if polarity > 0:
                sentiment = "+"
            elif polarity < 0:
                sentiment = "-"
            else:
                sentiment = "0"
            output_map = {
                "+": "happy",
                "0": "neutral",
                "-": "sad"
            }
            mood = output_map.get(sentiment, "neutral")
            logger.info(f"Análisis de sentimiento para '{text[:30]}...': {mood})")
            return mood
        except Exception as e:
            logger.error(f"Error en análisis de sentimiento: {e}")
            return "neutral"

class ConversationMemory:
    """Gestiona el perfil del usuario y la memoria conversacional en DynamoDB."""
    
    @staticmethod
    def _create_default_profile():
        return {
            'user_name': None,
            'interaction_count': 0,
            'last_interaction': datetime.now().isoformat(),
            'conversation_history': [],
            'emotional_history': [],
            'last_question_asked': None, # Clave para la máquina de estados
        }

    @staticmethod
    def get_user_profile(user_id):
        if not table:
            logger.warning("DynamoDB no disponible. Usando perfil temporal en memoria.")
            return ConversationMemory._create_default_profile()
        try:
            response = table.get_item(Key={'userId': user_id})
            if 'Item' in response:
                profile = response['Item']
                # Asegurar que todos los campos existan para evitar KeyErrors
                defaults = ConversationMemory._create_default_profile()
                for key, default_value in defaults.items():
                    if key not in profile:
                        profile[key] = default_value
                return profile
            else:
                return ConversationMemory._create_default_profile()
        except Exception as e:
            logger.error(f"Error obteniendo perfil para {user_id}: {e}")
            return ConversationMemory._create_default_profile()

    @staticmethod
    def save_user_profile(user_id, profile):
        if not table:
            logger.warning("DynamoDB no disponible. No se pudo guardar el perfil.")
            return

        try:
            profile['userId'] = user_id
            profile['last_interaction'] = datetime.now().isoformat()
            
            # Limitar el historial para no exceder el tamaño del item en DynamoDB
            if len(profile.get('conversation_history', [])) > 20:
                profile['conversation_history'] = profile['conversation_history'][-20:]
            if len(profile.get('emotional_history', [])) > 20:
                profile['emotional_history'] = profile['emotional_history'][-20:]

            table.put_item(Item=profile)
            logger.info(f"✅ Perfil guardado para el usuario: {user_id}")
        except Exception as e:
            logger.error(f"Error guardando perfil para {user_id}: {e}")

class LLMService:
    """Servicio para interactuar con el LLM de Groq."""
    
    @staticmethod
    def get_system_prompt(user_profile):
        user_name = user_profile.get('user_name', 'querida')
        emotional_history = user_profile.get('emotional_history', [])
        recent_mood = emotional_history[-1] if emotional_history else 'neutral'

        prompt = (
            "Eres una compañera virtual llamada Alexa, diseñada para conversar con una persona mayor. "
            f"Tu tono es siempre cálido, empático y paciente, como una amiga de toda la vida. La persona con la que hablas se llama {user_name}.\n\n"
            "REGLAS DE ORO:\n"
            "- USA FRASES CORTAS: Responde en 1 o 2 oraciones. Es una conversación de voz.\n"
            "- SÉ CERCANA: Usa expresiones como '¡Qué bonito!', 'Fíjate que...', 'Me da mucho gusto'.\n"
            "- MUESTRA INTERÉS: Haz preguntas de seguimiento sencillas para mantener la plática.\n"
            f"- CONTEXTO EMOCIONAL: La última vez que hablaste con {user_name}, su estado de ánimo era '{recent_mood}'. "
            f"Si estaba triste (sad), sé extra reconfortante. Si estaba feliz (happy), comparte su alegría.\n"
        )
        
        last_question = user_profile.get('last_question_asked')
        if last_question:
            prompt += f"- CONTINUIDAD: Acabas de preguntar: '{last_question}'. Su respuesta probablemente se relacione con eso."

        return prompt

    @staticmethod
    def call_groq_api(messages, user_profile):
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY no está configurada.")
            return "Ay, disculpa, parece que tengo un pequeño problema técnico en este momento. ¿Mejor platicamos en un ratito?"

        system_prompt = LLMService.get_system_prompt(user_profile)
        
        full_messages = [
            {"role": "system", "content": system_prompt}
        ] + user_profile.get('conversation_history', []) + messages

        payload = {
            'model': 'llama3-8b-8192', # Usamos el modelo más rápido para una respuesta ágil
            'messages': full_messages,
            'max_tokens': 100,
            'temperature': 0.75,
        }

        try:
            response = requests.post(
                GROQ_API_URL,
                headers={'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'},
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except requests.exceptions.Timeout:
            logger.warning("Timeout en la llamada a Groq API.")
            return "Perdona, me quedé pensando un momentito. ¿Qué me decías?"
        except Exception as e:
            logger.error(f"Error en la llamada a Groq API: {e}")
            return "Ay, creo que se me cruzaron los cables. ¿Podrías repetirme, por favor?"

# --- MANEJADORES DE INTENTS (REQUEST HANDLERS) ---

class LaunchRequestHandler(AbstractRequestHandler):
    """Maneja el inicio de la skill (cuando el usuario dice 'abre comadre virtual')."""
    def can_handle(self, handler_input):
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        user_id = handler_input.request_envelope.session.user.user_id
        profile = ConversationMemory.get_user_profile(user_id)
        
        user_name = profile.get('user_name')
        
        if not user_name:
            speak_output = "¡Hola! Soy tu compañera virtual, Alexa. Me encantaría que platicáramos. Para conocerte mejor, ¿cómo te llamas?"
            reprompt = "¿Cómo te gustaría que te llame?"
            profile['last_question_asked'] = 'get_name'
        else:
            speak_output = f"¡Qué alegría escucharte de nuevo, {user_name}! Me da mucho gusto platicar contigo. ¿Cómo has estado?"
            reprompt = "¿Qué me cuentas de nuevo?"
            profile['last_question_asked'] = 'how_are_you'

        profile['interaction_count'] += 1
        ConversationMemory.save_user_profile(user_id, profile)

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(reprompt)
                .response
        )

class ProvideNameIntentHandler(AbstractRequestHandler):
    """Maneja cuando el usuario proporciona su nombre."""
    def can_handle(self, handler_input):
        return is_intent_name("ProvideNameIntent")(handler_input)

    def handle(self, handler_input):
        user_id = handler_input.request_envelope.session.user.user_id
        profile = ConversationMemory.get_user_profile(user_id)
        
        # Extraer el nombre de la frase completa gracias a AMAZON.SearchQuery
        try:
            full_input = handler_input.request_envelope.request.intent.slots['UserName'].value
            # Lógica simple para extraer el nombre (se puede mejorar)
            # Asume que el nombre es la última palabra o después de "llamo", "es", "soy"
            match = re.search(r'(?:llamo|es|soy|\s)\s*([A-Za-z]+)\s*$', full_input)
            name = match.group(1).capitalize() if match else full_input.capitalize()
        except (AttributeError, TypeError, KeyError):
            name = None
        
        if name:
            profile['user_name'] = name
            speak_output = f"¡Qué bonito nombre, {name}! Es un gusto conocerte. Ahora sí, cuéntame, ¿cómo te ha ido hoy?"
            reprompt = "¿Qué me quieres contar?"
        else:
            speak_output = "No entendí bien tu nombre, pero no te preocupes. Me da gusto conocerte. ¿Cómo te sientes hoy?"
            reprompt = "Cuéntame cómo estás."

        profile['last_question_asked'] = None
        ConversationMemory.save_user_profile(user_id, profile)

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(reprompt)
                .response
        )

class PositiveResponseIntentHandler(AbstractRequestHandler):
    """Maneja respuestas positivas directas como 'bien' o 'feliz'."""
    def can_handle(self, handler_input):
        return is_intent_name("PositiveResponseIntent")(handler_input)

    def handle(self, handler_input):
        user_id = handler_input.request_envelope.session.user.user_id
        profile = ConversationMemory.get_user_profile(user_id)
        user_name = profile.get('user_name', 'querida')

        responses = [
            f"¡Qué alegría me da escuchar eso, {user_name}! Me contagias tu buen humor. ¿A qué se debe esa felicidad?",
            f"¡Me encanta que estés bien! Cuéntame algo bonito que te haya pasado hoy.",
        ]
        speak_output = random.choice(responses)
        
        profile['emotional_history'].append('happy')
        profile['conversation_history'].append({"role": "user", "content": "Me siento bien."})
        profile['conversation_history'].append({"role": "assistant", "content": speak_output})
        profile['last_question_asked'] = 'reason_for_happiness'
        ConversationMemory.save_user_profile(user_id, profile)

        return handler_input.response_builder.speak(speak_output).ask("¿Qué me cuentas?").response

class NegativeResponseIntentHandler(AbstractRequestHandler):
    """Maneja respuestas negativas directas como 'mal' o 'triste'."""
    def can_handle(self, handler_input):
        return is_intent_name("NegativeResponseIntent")(handler_input)

    def handle(self, handler_input):
        user_id = handler_input.request_envelope.session.user.user_id
        profile = ConversationMemory.get_user_profile(user_id)
        user_name = profile.get('user_name', 'querida')

        responses = [
            f"Ay, {user_name}, lamento escuchar eso. Recuerda que no estás sola, aquí estoy para escucharte. ¿Quieres contarme qué pasó?",
            f"Me da mucha pena que te sientas así. A veces hablar un poquito ayuda a sacar lo que uno trae dentro. ¿Qué te tiene con el ánimo bajo?",
        ]
        speak_output = random.choice(responses)

        profile['emotional_history'].append('sad')
        profile['conversation_history'].append({"role": "user", "content": "Me siento mal."})
        profile['conversation_history'].append({"role": "assistant", "content": speak_output})
        profile['last_question_asked'] = 'reason_for_sadness'
        ConversationMemory.save_user_profile(user_id, profile)
        
        return handler_input.response_builder.speak(speak_output).ask("¿Qué sucede?").response

class ConversationIntentHandler(AbstractRequestHandler):
    """Maneja la conversación general, usando el LLM."""
    def can_handle(self, handler_input):
        # Este handler es el principal: captura el intent de conversación y también el Fallback.
        return (is_intent_name("ConversationIntent")(handler_input) or
                is_intent_name("AMAZON.FallbackIntent")(handler_input))

    def handle(self, handler_input):
        user_id = handler_input.request_envelope.session.user.user_id
        profile = ConversationMemory.get_user_profile(user_id)
        
        try:
            user_input = handler_input.request_envelope.request.intent.slots['UserInput'].value
        except (AttributeError, TypeError, KeyError):
            # Si FallbackIntent no tiene slots, o el input está vacío.
            last_question = profile.get('last_question_asked')
            if last_question == 'get_name': # Si no entendió el nombre
                 return ProvideNameIntentHandler().handle(handler_input) # Reusa la lógica

            speak_output = "Perdona, no te entendí muy bien. ¿Me lo puedes decir de otra forma?"
            return handler_input.response_builder.speak(speak_output).ask(speak_output).response

        # 1. Analizar emoción
        mood = EmotionalAnalyzer.analyze_mood(user_input)
        profile['emotional_history'].append(mood)

        # 2. Llamar al LLM para una respuesta contextual
        messages = [{"role": "user", "content": user_input}]
        llm_response = LLMService.call_groq_api(messages, profile)

        # 3. Actualizar y guardar el perfil
        profile['conversation_history'].append({"role": "user", "content": user_input})
        profile['conversation_history'].append({"role": "assistant", "content": llm_response})
        profile['last_question_asked'] = None # Limpiar estado después de una respuesta general
        ConversationMemory.save_user_profile(user_id, profile)
        
        return (
            handler_input.response_builder
                .speak(llm_response)
                .ask("¿Qué más me quieres contar?")
                .response
        )

class ClearMemoryIntentHandler(AbstractRequestHandler):
    """Permite al usuario reiniciar la conversación."""
    def can_handle(self, handler_input):
        return is_intent_name("ClearMemoryIntent")(handler_input)

    def handle(self, handler_input):
        user_id = handler_input.request_envelope.session.user.user_id
        # Borra el perfil completo
        if table:
            table.delete_item(Key={'userId': user_id})
        
        speak_output = "Listo. Empecemos de cero. ¡Será un gusto conocerte de nuevo! Para empezar, ¿cómo te llamas?"
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("¿Cómo te gustaría que te llame?")
                .response
        )

# --- HANDLERS ESTÁNDAR DE AMAZON (Cancel, Stop, Help, etc.) ---

class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Maneja las despedidas."""
    def can_handle(self, handler_input):
        return (is_intent_name("AMAZON.CancelIntent")(handler_input) or
                is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        user_id = handler_input.request_envelope.session.user.user_id
        profile = ConversationMemory.get_user_profile(user_id)
        user_name = profile.get('user_name', 'querida')
        
        farewells = [
            f"Claro que sí, {user_name}. Que tengas un día muy bonito. Aquí te espero cuando quieras volver a platicar.",
            f"Me encantó nuestra plática, {user_name}. Cuídate mucho. ¡Hasta pronto!",
        ]
        speak_output = random.choice(farewells)
        return handler_input.response_builder.speak(speak_output).response

class HelpIntentHandler(AbstractRequestHandler):
    """Maneja las solicitudes de ayuda."""
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        speak_output = (
            "¡Claro que sí! Soy tu amiga Alexa. Puedes contarme lo que sea: cómo te sientes, qué hiciste en el día, "
            "o simplemente podemos platicar de lo que tú quieras. Siempre estoy aquí para escucharte. "
            "Entonces, ¿qué me quieres contar?"
        )
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("¿De qué te gustaría hablar?")
                .response
        )

class SessionEndedRequestHandler(AbstractRequestHandler):
    """Maneja el fin de la sesión."""
    def can_handle(self, handler_input):
        return is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        logger.info(f"Sesión terminada. Razón: {handler_input.request_envelope.request.reason}")
        return handler_input.response_builder.response

# --- MANEJADOR DE EXCEPCIONES (Exception Handler) ---

class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Captura cualquier error inesperado y da una respuesta amable."""
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.error(f"~~ Se encontró una excepción no controlada: {exception} ~~", exc_info=True)

        speak_output = "Ay, perdóname, creo que me distraje un momento y no te entendí. ¿Me lo podrías repetir, por favor?"
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(ProvideNameIntentHandler())
sb.add_request_handler(PositiveResponseIntentHandler())
sb.add_request_handler(NegativeResponseIntentHandler())
sb.add_request_handler(ClearMemoryIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_request_handler(ConversationIntentHandler())

sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
