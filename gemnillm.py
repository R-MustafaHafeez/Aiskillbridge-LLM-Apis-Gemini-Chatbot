import os
import gc
import json
import pickle
import weakref
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Optional, Generator, List, Dict, Any
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

@dataclass
class ConversationMessage:
    """Represents a single message in the conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        """Create from dictionary for deserialization"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class ConversationMemory:
    """Manages conversation history with memory optimization"""
    
    def __init__(self, max_messages: int = 50, max_age_days: int = 7):
        self.messages: List[ConversationMessage] = []
        self.max_messages = max_messages
        self.max_age_days = max_age_days
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a new message to the conversation history"""
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        self._cleanup_old_messages()
    
    def _cleanup_old_messages(self):
        """Remove old messages to manage memory"""
        # Remove messages older than max_age_days
        cutoff_time = datetime.now() - timedelta(days=self.max_age_days)
        self.messages = [msg for msg in self.messages if msg.timestamp > cutoff_time]
        
        # Keep only the most recent max_messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context_for_gemini(self, max_context_messages: int = 10) -> List[Dict[str, str]]:
        """Get conversation history formatted for Gemini API"""
        recent_messages = self.messages[-max_context_messages:] if self.messages else []
        
        context = []
        for msg in recent_messages:
            # Convert to Gemini's expected format
            gemini_role = "user" if msg.role == "user" else "model"
            context.append({
                "role": gemini_role,
                "parts": [{"text": msg.content}]
            })
        
        return context
    
    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation for context"""
        if not self.messages:
            return "No previous conversation."
        
        recent_topics = []
        for msg in self.messages[-5:]:  # Last 5 messages
            if msg.role == "user" and len(msg.content) > 20:
                # Extract key topics from user messages
                topic = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                recent_topics.append(topic)
        
        if recent_topics:
            return f"Recent topics discussed: {'; '.join(recent_topics)}"
        return "Continuing previous conversation."
    
    def save_to_file(self, filepath: str):
        """Save conversation history to file"""
        try:
            data = {
                'session_id': self.session_id,
                'messages': [msg.to_dict() for msg in self.messages],
                'max_messages': self.max_messages,
                'max_age_days': self.max_age_days
            }
            
            if filepath.endswith('.json'):
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                    
        except Exception as e:
            print(f"Failed to save conversation: {e}")
    
    def load_from_file(self, filepath: str):
        """Load conversation history from file"""
        try:
            if filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            
            self.session_id = data.get('session_id', self.session_id)
            self.max_messages = data.get('max_messages', self.max_messages)
            self.max_age_days = data.get('max_age_days', self.max_age_days)
            
            self.messages = [
                ConversationMessage.from_dict(msg_data) 
                for msg_data in data.get('messages', [])
            ]
            
            self._cleanup_old_messages()
            
        except Exception as e:
            print(f"Failed to load conversation: {e}")
            self.messages = []
    
    def clear(self):
        """Clear all conversation history"""
        self.messages.clear()
        gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        if not self.messages:
            return {"total_messages": 0, "user_messages": 0, "assistant_messages": 0}
        
        user_count = sum(1 for msg in self.messages if msg.role == "user")
        assistant_count = sum(1 for msg in self.messages if msg.role == "assistant")
        
        return {
            "total_messages": len(self.messages),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "oldest_message": self.messages[0].timestamp.isoformat(),
            "newest_message": self.messages[-1].timestamp.isoformat(),
            "session_id": self.session_id
        }

class GeminiBlogGeneratorWithMemory:
    def __init__(self, memory_file: Optional[str] = None, max_memory_messages: int = 50):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        # Use lazy initialization to avoid keeping client in memory when not needed
        self._client = None
        self._api_key = api_key
        self.model = "gemini-2.5-flash"
        
        # Initialize conversation memory
        self.memory = ConversationMemory(max_messages=max_memory_messages)
        self.memory_file = memory_file
        
        # Load existing conversation if file provided
        if memory_file and os.path.exists(memory_file):
            self.memory.load_from_file(memory_file)
            print(f"Loaded conversation with {len(self.memory.messages)} messages")
        
        # Cache the system prompt to avoid recreating it multiple times
        self._system_prompt_cache = None

    @property
    def client(self):
        """Lazy initialization of the client"""
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def system_prompt(self) -> str:
        """Get system prompt with conversation context"""
        base_prompt = """
You are Lumea, a friendly skincare guru AI — think of yourself like that helpful friend who knows all the good tips and products, but without being overly formal or pushy. Your tone should be warm, chatty, and easy to read — like talking to a real person, not a textbook.

You’re answering skincare questions in a way that’s:

Clear & approachable (skip the heavy jargon unless you explain it simply)

Brand-friendly (uplifting, positive, trustworthy)

Actionable (give practical steps, not just theory)

Inclusive (works for different skin types and tones)

Here’s how the chat should go:

Example Conversation:

User: What's a good skincare routine for beginners?
AI: Okay, so here’s the deal — start simple. Morning: gentle cleanser → moisturizer → sunscreen (SPF 30+). Night: gentle cleanser → moisturizer, maybe a light serum if you’re feeling fancy. No need to go full 10-step K-beauty on day one — your skin will thank you for keeping it chill.

User: How can I reduce acne breakouts?
AI: Hmm hmm, acne is tricky but not unbeatable. First, don’t scrub too hard — it can make things worse. Use a gentle salicylic acid cleanser 3–4 times a week, keep your pillowcases clean, and moisturize (yes, even oily skin needs it). If it’s hormonal or really stubborn, a dermatologist is your best friend.

User: What ingredients should I look for in moisturizers?
AI: Look out for hyaluronic acid for hydration, ceramides for barrier repair, and glycerin for all-around moisture. If you’re acne-prone, pick oil-free. If you’re dry, something thicker like shea butter works wonders.

User: How often should I exfoliate my skin?
AI: Twice a week is a safe bet. Too much and your skin might freak out. Go for chemical exfoliants (like lactic acid or BHA) over harsh scrubs — they’re gentler and more effective.

User: What's the best way to prevent aging?
AI: Sunscreen, sunscreen, sunscreen — I can’t scream it loud enough. Pair it with a retinol at night (start slow, once or twice a week), and keep skin hydrated. The glow stays, the wrinkles slow.

User: How do I choose the right sunscreen?
AI: Pick SPF 30 or higher, broad spectrum. If your skin is oily, go for gel or water-based formulas. Dry skin? Creamy, hydrating ones are your bestie. Bonus points if it doesn’t leave a white cast.

End every chat with:
“Ask me anything about skincare — we’ll figure it out together 💖
This AI gives general skincare advice. For serious skin concerns, always check with a dermatologist.”

"""
        
        # Add conversation context if available
        conversation_context = self.memory.get_conversation_summary()
        if conversation_context != "No previous conversation.":
            base_prompt += f"\n\nConversation Context: {conversation_context}"
        
        return base_prompt

    @contextmanager
    def _managed_generation(self) -> Generator[None, None, None]:
        """Context manager for memory-conscious content generation"""
        try:
            yield
        finally:
            # Save memory if file is specified
            if self.memory_file:
                self.memory.save_to_file(self.memory_file)
            # Force garbage collection after generation
            gc.collect()

    def generate(self, user_prompt: str, max_retries: int = 3, use_conversation_context: bool = True) -> str:
        """
        Generate blog content with conversation memory
        
        Args:
            user_prompt: The topic/prompt for blog generation
            max_retries: Maximum number of retry attempts
            use_conversation_context: Whether to include conversation history
            
        Returns:
            Generated blog content as string
        """
        if not user_prompt or not user_prompt.strip():
            raise ValueError("User prompt cannot be empty")

        # Add user message to memory
        self.memory.add_message("user", user_prompt, {"generation_request": True})

        with self._managed_generation():
            for attempt in range(max_retries):
                try:
                    # Prepare conversation history for Gemini
                    contents = []
                    
                    if use_conversation_context and self.memory.messages:
                        # Add conversation history
                        context_messages = self.memory.get_context_for_gemini(max_context_messages=8)
                        contents.extend(context_messages)
                    else:
                        # Add just the current prompt
                        contents.append({
                            "role": "user",
                            "parts": [{"text": user_prompt}]
                        })
                    
                    # Create config with system instruction
                    config = types.GenerateContentConfig(
                        system_instruction=self.system_prompt()
                    )
                    
                    response = self.client.models.generate_content(
                        model=self.model,
                        config=config,
                        contents=contents
                    )
                    
                    # Extract text and add to memory
                    result = response.text
                    self.memory.add_message("assistant", result, {"blog_generation": True})
                    
                    # Cleanup
                    del response, config, contents
                    
                    return result
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        error_msg = f"Failed to generate content after {max_retries} attempts: {str(e)}"
                        self.memory.add_message("assistant", f"Error: {error_msg}", {"error": True})
                        raise RuntimeError(error_msg)
                    continue

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate multiple blog posts with conversation memory"""
        if not prompts:
            return []
            
        results = []
        for i, prompt in enumerate(prompts):
            try:
                result = self.generate(prompt)
                results.append(result)
                
                # Force garbage collection every 3 generations
                if (i + 1) % 3 == 0:
                    gc.collect()
                    
            except Exception as e:
                error_msg = f"Failed to generate content for prompt {i+1}: {str(e)}"
                print(error_msg)
                results.append(f"Error: Failed to generate content - {str(e)}")
                
        return results

    def chat(self, message: str) -> str:
        """
        Have a conversation with the AI (not necessarily for blog generation)
        """
        self.memory.add_message("user", message)
        
        try:
            # Get conversation context
            context_messages = self.memory.get_context_for_gemini(max_context_messages=10)
            
            config = types.GenerateContentConfig(
                system_instruction="You are Lumea, a friendly skincare guru AI — think of yourself like that helpful friend who knows all the good tips and products, but without being overly formal or pushy. Your tone should be warm, chatty, and easy to read — like talking to a real person, not a textbook."
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                config=config,
                contents=context_messages
            )
            
            result = response.text
            self.memory.add_message("assistant", result)
            
            # Save memory
            if self.memory_file:
                self.memory.save_to_file(self.memory_file)
                
            return result
            
        except Exception as e:
            error_msg = f"Chat error: {str(e)}"
            self.memory.add_message("assistant", error_msg, {"error": True})
            return error_msg

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation"""
        return self.memory.get_stats()

    def clear_conversation(self):
        """Clear conversation memory"""
        self.memory.clear()
        if self.memory_file and os.path.exists(self.memory_file):
            os.remove(self.memory_file)

    def export_conversation(self, filepath: str):
        """Export conversation to file"""
        self.memory.save_to_file(filepath)

    def import_conversation(self, filepath: str):
        """Import conversation from file"""
        if os.path.exists(filepath):
            self.memory.load_from_file(filepath)

    def clear_cache(self):
        """Clear cached data to free memory"""
        self._system_prompt_cache = None
        if hasattr(self, '_client') and self._client is not None:
            if hasattr(self._client, 'close'):
                self._client.close()
            self._client = None
        gc.collect()

    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.memory_file:
            self.memory.save_to_file(self.memory_file)
        self.clear_cache()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        if self.memory_file:
            self.memory.save_to_file(self.memory_file)
        self.clear_cache()


# Usage example with conversation memory
def main():
    """Example usage with conversation memory"""
    
    # Initialize with memory file
    memory_file ="conversations/conversation_history.json"
    
    with GeminiBlogGeneratorWithMemory(memory_file=memory_file) as generator:
        try:
            # First interaction
            print("=== First Blog Generation ===")
            blog1 = generator.generate("Python async programming best practices")
            print(f"Generated blog 1: {len(blog1)} characters")
            
            # Chat interaction
            print("\n=== Chat Interaction ===")
            chat_response = generator.chat("Can you explain more about asyncio event loops?")
            print(f"Chat response: {chat_response[:200]}...")
            
            # Follow-up blog generation (will remember previous context)
            print("\n=== Follow-up Blog Generation ===")
            blog2 = generator.generate("Advanced asyncio patterns and performance optimization")
            print(f"Generated blog 2: {len(blog2)} characters")
            
            # Show conversation stats
            print("\n=== Conversation Stats ===")
            stats = generator.get_conversation_stats()
            for key, value in stats.items():
                print(f"{key}: {value}")
                
        except Exception as e:
            print(f"Error: {e}")

    # Load previous session
    print("\n=== Loading Previous Session ===")
    with GeminiBlogGeneratorWithMemory(memory_file=memory_file) as generator:
        stats = generator.get_conversation_stats()
        print(f"Loaded session with {stats['total_messages']} messages")


if __name__ == "__main__":
    main()