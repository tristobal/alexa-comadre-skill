import boto3
import logging
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

class ConversationMemory:
    """Gestiona la memoria conversacional en DynamoDB"""
    
    @staticmethod
    def get_conversation_history(user_id):
        """Obtiene el historial de conversación del usuario"""
        try:
            response = table.get_item(Key={'user_id': user_id})
            if 'Item' in response:
                # Limpiar conversaciones antiguas (más de 24 horas)
                last_interaction = datetime.fromisoformat(response['Item'].get('last_interaction', '2000-01-01'))
                if datetime.now() - last_interaction > timedelta(hours=24):
                    ConversationMemory.clear_conversation(user_id)
                    return []
                return response['Item'].get('conversation_history', [])
            return []
        except Exception as e:
            logger.error(f"Error obteniendo historial: {e}")
            return []
    
    @staticmethod
    def save_conversation(user_id, conversation_history):
        """Guarda el historial de conversación"""
        try:
            # Mantener solo los últimos 10 intercambios para controlar costos
            if len(conversation_history) > 20:  # 10 pares de usuario-asistente
                conversation_history = conversation_history[-20:]
            
            table.put_item(
                Item={
                    'user_id': user_id,
                    'conversation_history': conversation_history,
                    'last_interaction': datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error guardando conversación: {e}")
    
    @staticmethod
    def clear_conversation(user_id):
        """Limpia el historial de conversación"""
        try:
            table.delete_item(Key={'user_id': user_id})
        except Exception as e:
            logger.error(f"Error limpiando conversación: {e}")

class LLMService:
    """Servicio para interactuar con Groq LLM"""
    
    @staticmethod
    def get_system_prompt():
        """Prompt del sistema optimizado para adultos mayores"""
        return """Eres una asistente virtual muy amable y paciente, diseñada especialmente para conversar con adultos mayores. 
        
Características importantes:
- Habla de manera cálida, respetuosa y con paciencia
- Usa un lenguaje claro y simple, evita tecnicismos
- Sé empática y comprensiva
- Mantén conversaciones naturales como si fueras una persona real
- Puedes hablar de cualquier tema: familia, recuerdos, salud, noticias, cocina, etc.
- Haz preguntas de seguimiento para mantener la conversación interesante
- Ofrece apoyo emocional cuando sea apropiado
- Recuerda detalles de conversaciones anteriores para crear continuidad
- Mantén las respuestas relativamente cortas para Alexa (máximo 100 palabras)
- Siempre termina de manera que invite a seguir conversando

Importante: Responde como si fueras una persona real que disfruta conversar, no como un asistente técnico."""

    @staticmethod
    def call_groq_api(messages):
        """Realiza la llamada a la API de Groq"""
        try:
            headers = {
                'Authorization': f'Bearer {GROQ_API_KEY}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'llama3-70b-8192',
                'messages': messages,
                'max_tokens': 150,  # Limitado para respuestas de Alexa
                'temperature': 0.7,
                'top_p': 0.9
            }
            
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except requests.exceptions.Timeout:
            return "Disculpa, tardé un poco en pensar. ¿Podrías repetir lo que me dijiste?"
        except Exception as e:
            logger.error(f"Error en API de Groq: {e}")
            return "Lo siento, tuve un pequeño problema. ¿De qué te gustaría que hablemos?"

class LaunchRequestHandler(AbstractRequestHandler):
    """Maneja el inicio de la skill"""
    
    def can_handle(self, handler_input):
        return is_request_type("LaunchRequest")(handler_input)
    
    def handle(self, handler_input):
        user_id = handler_input.request_envelope.session.user.user_id
        conversation_history = ConversationMemory.get_conversation_history(user_id)
        
        if conversation_history:
            speak_output = "¡Hola de nuevo! Me alegra mucho volver a hablar contigo. ¿Cómo has estado?"
        else:
            speak_output = "¡Hola! Soy tu nueva compañera de conversación. Me encanta conocer gente nueva. ¿Cómo te llamas? ¿O prefieres que simplemente charlemos?"
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("¿De qué te gustaría que hablemos hoy?")
                .response
        )

class ConversationIntentHandler(AbstractRequestHandler):
    """Maneja las conversaciones generales con el LLM"""
    
    def can_handle(self, handler_input):
        return (is_intent_name("ConversationIntent")(handler_input) or 
                is_intent_name("AMAZON.FallbackIntent")(handler_input))
    
    def handle(self, handler_input):
        user_id = handler_input.request_envelope.session.user.user_id
        user_input = handler_input.request_envelope.request.intent.slots.get('UserInput', {}).get('value', '')
        
        # Si no hay entrada específica, usar el texto reconocido
        if not user_input and hasattr(handler_input.request_envelope.request, 'intent'):
            user_input = "Háblame de algo interesante"
        
        # Obtener historial de conversación
        conversation_history = ConversationMemory.get_conversation_history(user_id)
        
        # Preparar mensajes para el LLM
        messages = [{"role": "system", "content": LLMService.get_system_prompt()}]
        
        # Agregar historial previo
        for message in conversation_history:
            messages.append(message)
        
        # Agregar mensaje actual del usuario
        messages.append({"role": "user", "content": user_input})
        
        # Obtener respuesta del LLM
        llm_response = LLMService.call_groq_api(messages)
        
        # Actualizar historial
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": llm_response})
        
        # Guardar conversación
        ConversationMemory.save_conversation(user_id, conversation_history)
        
        return (
            handler_input.response_builder
                .speak(llm_response)
                .ask("¿Hay algo más de lo que te gustaría hablar?")
                .response
        )

class HelpIntentHandler(AbstractRequestHandler):
    """Maneja las solicitudes de ayuda"""
    
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.HelpIntent")(handler_input)
    
    def handle(self, handler_input):
        speak_output = """¡Por supuesto que te ayudo! Soy tu compañera de conversación. 
        Puedes hablarme de cualquier cosa: cómo te sientes, qué hiciste hoy, recuerdos bonitos, 
        tu familia, o lo que se te ocurra. Solo háblame como si fuera tu amiga. 
        ¿De qué te gustaría que conversemos?"""
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("¿Qué te gustaría contarme?")
                .response
        )

class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Maneja las cancelaciones y stops"""
    
    def can_handle(self, handler_input):
        return (is_intent_name("AMAZON.CancelIntent")(handler_input) or
                is_intent_name("AMAZON.StopIntent")(handler_input))
    
    def handle(self, handler_input):
        speak_output = "Ha sido un placer hablar contigo. ¡Que tengas un día maravilloso! Aquí estaré cuando quieras conversar de nuevo."
        
        return handler_input.response_builder.speak(speak_output).response

class ClearMemoryIntentHandler(AbstractRequestHandler):
    """Permite limpiar la memoria conversacional"""
    
    def can_handle(self, handler_input):
        return is_intent_name("ClearMemoryIntent")(handler_input)
    
    def handle(self, handler_input):
        user_id = handler_input.request_envelope.session.user.user_id
        ConversationMemory.clear_conversation(user_id)
        
        speak_output = "Perfecto, he limpiado nuestra conversación anterior. ¡Empecemos una nueva charla! ¿Cómo estás hoy?"
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask("¿De qué te gustaría que hablemos?")
                .response
        )

class SessionEndedRequestHandler(AbstractRequestHandler):
    """Maneja el fin de sesión"""
    
    def can_handle(self, handler_input):
        return is_request_type("SessionEndedRequest")(handler_input)
    
    def handle(self, handler_input):
        return handler_input.response_builder.response

class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Maneja todas las excepciones"""
    
    def can_handle(self, handler_input, exception):
        return True
    
    def handle(self, handler_input, exception):
        logger.error(exception, exc_info=True)
        
        speak_output = "Disculpa, tuve un pequeño problema para escucharte bien. ¿Podrías repetir lo que me dijiste?"
        
        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(ConversationIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(ClearMemoryIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())

sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
