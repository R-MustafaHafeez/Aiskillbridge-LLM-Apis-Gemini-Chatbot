# import os
# import uuid
# import asyncio
# from datetime import datetime, timedelta
# from typing import Dict, Optional
# from contextlib import asynccontextmanager
# from fastapi import FastAPI, Request, HTTPException, Depends
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# import threading
# import weakref
# from gemnillm import GeminiBlogGeneratorWithMemory

# # Pydantic models for API requests/responses
# class PromptRequest(BaseModel):
#     user_prompt: str
#     user_id: Optional[str] = None
#     session_id: Optional[str] = None

# class ChatRequest(BaseModel):
#     message: str
#     user_id: Optional[str] = None
#     session_id: Optional[str] = None

# class UserSessionResponse(BaseModel):
#     user_id: str
#     session_id: str
#     message: str

# class ConversationStatsResponse(BaseModel):
#     user_id: str
#     session_id: str
#     stats: dict

# class SessionManager:
#     """Manages user sessions and their conversation memories"""
    
#     def __init__(self, cleanup_interval_minutes: int = 30, session_timeout_hours: int = 24):
#         self.sessions: Dict[str, Dict[str, GeminiBlogGeneratorWithMemory]] = {}
#         self.session_last_activity: Dict[str, Dict[str, datetime]] = {}
#         self.cleanup_interval = cleanup_interval_minutes
#         self.session_timeout = session_timeout_hours
#         self.lock = threading.RLock()
        
#         # Start cleanup task
#         self._start_cleanup_task()
    
#     def _start_cleanup_task(self):
#         """Start background task to cleanup inactive sessions"""
#         def cleanup_task():
#             while True:
#                 try:
#                     self._cleanup_inactive_sessions()
#                     threading.Event().wait(self.cleanup_interval * 60)  # Convert to seconds
#                 except Exception as e:
#                     print(f"Cleanup task error: {e}")
        
#         cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
#         cleanup_thread.start()
    
#     def _cleanup_inactive_sessions(self):
#         """Remove inactive sessions to free memory"""
#         cutoff_time = datetime.now() - timedelta(hours=self.session_timeout)
        
#         with self.lock:
#             users_to_remove = []
#             for user_id, user_sessions in self.sessions.items():
#                 sessions_to_remove = []
                
#                 for session_id, last_activity in self.session_last_activity.get(user_id, {}).items():
#                     if last_activity < cutoff_time:
#                         sessions_to_remove.append(session_id)
                
#                 # Remove inactive sessions
#                 for session_id in sessions_to_remove:
#                     if session_id in user_sessions:
#                         try:
#                             user_sessions[session_id].clear_cache()
#                             del user_sessions[session_id]
#                         except Exception as e:
#                             print(f"Error cleaning up session {session_id}: {e}")
                    
#                     if user_id in self.session_last_activity and session_id in self.session_last_activity[user_id]:
#                         del self.session_last_activity[user_id][session_id]
                
#                 # Remove user if no sessions left
#                 if not user_sessions:
#                     users_to_remove.append(user_id)
            
#             # Remove empty users
#             for user_id in users_to_remove:
#                 if user_id in self.sessions:
#                     del self.sessions[user_id]
#                 if user_id in self.session_last_activity:
#                     del self.session_last_activity[user_id]
            
#             if users_to_remove or any(sessions_to_remove for sessions_to_remove in [[] for _ in self.sessions.values()]):
#                 print(f"Cleaned up inactive sessions. Active users: {len(self.sessions)}")
    
#     def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> tuple[str, str, GeminiBlogGeneratorWithMemory]:
#         """Get existing session or create new one"""
#         if not user_id:
#             user_id = str(uuid.uuid4())
        
#         if not session_id:
#             session_id = str(uuid.uuid4())
        
#         with self.lock:
#             # Initialize user if doesn't exist
#             if user_id not in self.sessions:
#                 self.sessions[user_id] = {}
#                 self.session_last_activity[user_id] = {}
            
#             # Create session if doesn't exist
#             if session_id not in self.sessions[user_id]:
#                 memory_file = f"conversations/{user_id}_{session_id}.json"
#                 os.makedirs("conversations", exist_ok=True)
                
#                 generator = GeminiBlogGeneratorWithMemory(
#                     memory_file=memory_file,
#                     max_memory_messages=100
#                 )
#                 self.sessions[user_id][session_id] = generator
            
#             # Update last activity
#             self.session_last_activity[user_id][session_id] = datetime.now()
            
#             return user_id, session_id, self.sessions[user_id][session_id]
    
#     def get_session(self, user_id: str, session_id: str) -> Optional[GeminiBlogGeneratorWithMemory]:
#         """Get existing session"""
#         with self.lock:
#             if user_id in self.sessions and session_id in self.sessions[user_id]:
#                 self.session_last_activity[user_id][session_id] = datetime.now()
#                 return self.sessions[user_id][session_id]
#             return None
    
#     def list_user_sessions(self, user_id: str) -> list[str]:
#         """List all sessions for a user"""
#         with self.lock:
#             return list(self.sessions.get(user_id, {}).keys())
    
#     def delete_session(self, user_id: str, session_id: str) -> bool:
#         """Delete a specific session"""
#         with self.lock:
#             if user_id in self.sessions and session_id in self.sessions[user_id]:
#                 try:
#                     self.sessions[user_id][session_id].clear_cache()
#                     del self.sessions[user_id][session_id]
                    
#                     if user_id in self.session_last_activity and session_id in self.session_last_activity[user_id]:
#                         del self.session_last_activity[user_id][session_id]
                    
#                     # Remove user if no sessions left
#                     if not self.sessions[user_id]:
#                         del self.sessions[user_id]
#                         if user_id in self.session_last_activity:
#                             del self.session_last_activity[user_id]
                    
#                     # Delete conversation file
#                     memory_file = f"conversations/{user_id}_{session_id}.json"
#                     if os.path.exists(memory_file):
#                         os.remove(memory_file)
                    
#                     return True
#                 except Exception as e:
#                     print(f"Error deleting session: {e}")
#                     return False
#             return False
    
#     def get_stats(self) -> dict:
#         """Get system statistics"""
#         with self.lock:
#             total_sessions = sum(len(user_sessions) for user_sessions in self.sessions.values())
#             return {
#                 "total_users": len(self.sessions),
#                 "total_sessions": total_sessions,
#                 "sessions_per_user": {user_id: len(sessions) for user_id, sessions in self.sessions.items()}
#             }

# # Global session manager
# session_manager = SessionManager()

# # Lifespan management
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup
#     print("🚀 FastAPI Blog Generator with Multi-User Chat Starting...")
#     print(f"💾 Conversation files will be stored in: {os.path.abspath('conversations')}")
#     yield
#     # Shutdown
#     print("🔄 Cleaning up sessions...")
#     # Cleanup all sessions
#     for user_id in list(session_manager.sessions.keys()):
#         for session_id in list(session_manager.sessions[user_id].keys()):
#             session_manager.delete_session(user_id, session_id)
#     print("✅ Shutdown complete")

# # FastAPI app with lifespan
# app = FastAPI(
#     title="Multi-User Blog Generator API",
#     description="FastAPI application with multi-user chat memory for blog generation",
#     version="1.0.0",
#     lifespan=lifespan
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Helper function to extract user info
# async def get_user_session(request: PromptRequest) -> tuple[str, str, GeminiBlogGeneratorWithMemory]:
#     """Extract or create user session"""
#     return session_manager.get_or_create_session(request.user_id, request.session_id)

# async def get_chat_session(request: ChatRequest) -> tuple[str, str, GeminiBlogGeneratorWithMemory]:
#     """Extract or create chat session"""
#     return session_manager.get_or_create_session(request.user_id, request.session_id)

# # API Routes
# @app.get("/")
# def read_root():
#     return JSONResponse(content={
#         "message": "Multi-User Blog Generator API",
#         "version": "1.0.0",
#         "features": ["Blog Generation", "Multi-User Chat", "Session Management", "Conversation Memory"],
#         "endpoints": {
#             "generate": "POST /generate - Generate blog posts",
#             "chat": "POST /chat - Chat with AI",
#             "stats": "GET /stats/{user_id}/{session_id} - Get conversation stats",
#             "sessions": "GET /sessions/{user_id} - List user sessions",
#             "system_stats": "GET /system/stats - Get system statistics"
#         }
#     })

# @app.post("/generate")
# async def generate_blog(request: PromptRequest):
#     """Generate a blog post with conversation memory"""
#     try:
#         user_id, session_id, generator = await get_user_session(request)
        
#         # Generate blog content
#         response_text = generator.generate(
#             request.user_prompt,
#             use_conversation_context=True
#         )
        
#         return JSONResponse(content={
#             "response": response_text,
#             "user_id": user_id,
#             "session_id": session_id,
#             "timestamp": datetime.now().isoformat()
#         })
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# @app.post("/chat")
# async def chat_with_ai(request: ChatRequest):
#     """Chat with AI (non-blog generation)"""
#     try:
#         user_id, session_id, generator = await get_chat_session(request)
        
#         # Chat with AI
#         response_text = generator.chat(request.message)
        
#         return JSONResponse(content={
#             "response": response_text,
#             "user_id": user_id,
#             "session_id": session_id,
#             "timestamp": datetime.now().isoformat()
#         })
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

# @app.get("/stats/{user_id}/{session_id}")
# async def get_conversation_stats(user_id: str, session_id: str):
#     """Get conversation statistics for a specific session"""
#     generator = session_manager.get_session(user_id, session_id)
#     if not generator:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     stats = generator.get_conversation_stats()
#     return JSONResponse(content={
#         "user_id": user_id,
#         "session_id": session_id,
#         "stats": stats
#     })

# @app.get("/sessions/{user_id}")
# async def list_user_sessions(user_id: str):
#     """List all sessions for a user"""
#     sessions = session_manager.list_user_sessions(user_id)
#     return JSONResponse(content={
#         "user_id": user_id,
#         "sessions": sessions,
#         "session_count": len(sessions)
#     })

# @app.delete("/sessions/{user_id}/{session_id}")
# async def delete_user_session(user_id: str, session_id: str):
#     """Delete a specific session"""
#     success = session_manager.delete_session(user_id, session_id)
#     if success:
#         return JSONResponse(content={
#             "message": "Session deleted successfully",
#             "user_id": user_id,
#             "session_id": session_id
#         })
#     else:
#         raise HTTPException(status_code=404, detail="Session not found")

# @app.post("/sessions/{user_id}/{session_id}/clear")
# async def clear_session_conversation(user_id: str, session_id: str):
#     """Clear conversation history for a session"""
#     generator = session_manager.get_session(user_id, session_id)
#     if not generator:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     generator.clear_conversation()
#     return JSONResponse(content={
#         "message": "Conversation cleared successfully",
#         "user_id": user_id,
#         "session_id": session_id
#     })

# @app.get("/system/stats")
# async def get_system_stats():
#     """Get system-wide statistics"""
#     stats = session_manager.get_stats()
#     return JSONResponse(content={
#         "system_stats": stats,
#         "timestamp": datetime.now().isoformat()
#     })

# @app.post("/batch/generate")
# async def generate_batch_blogs(request: dict):
#     """Generate multiple blog posts in batch"""
#     try:
#         prompts = request.get("prompts", [])
#         user_id = request.get("user_id")
#         session_id = request.get("session_id")
        
#         if not prompts:
#             raise HTTPException(status_code=400, detail="No prompts provided")
        
#         user_id, session_id, generator = session_manager.get_or_create_session(user_id, session_id)
        
#         # Generate batch
#         results = generator.generate_batch(prompts)
        
#         return JSONResponse(content={
#             "results": results,
#             "user_id": user_id,
#             "session_id": session_id,
#             "generated_count": len(results),
#             "timestamp": datetime.now().isoformat()
#         })
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")

# # Health check endpoint
# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return JSONResponse(content={
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "system_stats": session_manager.get_stats()
#     })

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "main:app", 
#         host="0.0.0.0", 
#         port=8000, 
#         reload=True,
#         log_level="info"
#     )













# main.py
import os
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gemnillm import GeminiBlogGeneratorWithMemory

# === Pydantic models ===
class PromptRequest(BaseModel):
    user_prompt: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

# === Session manager ===
class SessionManager:
    def __init__(self, cleanup_interval_minutes: int = 30, session_timeout_hours: int = 24):
        self.sessions: Dict[str, Dict[str, GeminiBlogGeneratorWithMemory]] = {}
        self.session_last_activity: Dict[str, Dict[str, datetime]] = {}
        self.cleanup_interval = cleanup_interval_minutes
        self.session_timeout = session_timeout_hours
        self.lock = threading.RLock()
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        def cleanup_task():
            while True:
                try:
                    self._cleanup_inactive_sessions()
                    threading.Event().wait(self.cleanup_interval * 60)
                except Exception as e:
                    print(f"Cleanup task error: {e}")
        t = threading.Thread(target=cleanup_task, daemon=True)
        t.start()

    def _cleanup_inactive_sessions(self):
        cutoff = datetime.now() - timedelta(hours=self.session_timeout)
        with self.lock:
            users_to_remove = []
            for user_id, user_sessions in list(self.sessions.items()):
                sessions_to_remove = []
                for session_id, last_activity in list(self.session_last_activity.get(user_id, {}).items()):
                    if last_activity < cutoff:
                        sessions_to_remove.append(session_id)
                for session_id in sessions_to_remove:
                    if session_id in user_sessions:
                        try:
                            user_sessions[session_id].clear_cache()
                            del user_sessions[session_id]
                        except Exception as e:
                            print(f"Error cleaning up session {session_id}: {e}")
                    if user_id in self.session_last_activity and session_id in self.session_last_activity[user_id]:
                        del self.session_last_activity[user_id][session_id]
                if not user_sessions:
                    users_to_remove.append(user_id)
            for user_id in users_to_remove:
                self.sessions.pop(user_id, None)
                self.session_last_activity.pop(user_id, None)
            if users_to_remove:
                print(f"Cleaned up inactive sessions. Active users: {len(self.sessions)}")

    def get_or_create_session(self, user_id: Optional[str], session_id: Optional[str]) -> tuple[str, str, GeminiBlogGeneratorWithMemory]:
        if not user_id:
            user_id = str(uuid.uuid4())
        if not session_id:
            session_id = str(uuid.uuid4())

        with self.lock:
            if user_id not in self.sessions:
                self.sessions[user_id] = {}
                self.session_last_activity[user_id] = {}

            if session_id not in self.sessions[user_id]:
                os.makedirs("conversations", exist_ok=True)
                memory_file = f"conversations/{user_id}_{session_id}.json"
                generator = GeminiBlogGeneratorWithMemory(
                    memory_file=memory_file,
                    max_memory_messages=100
                )
                self.sessions[user_id][session_id] = generator

            self.session_last_activity[user_id][session_id] = datetime.now()
            return user_id, session_id, self.sessions[user_id][session_id]

    def get_session(self, user_id: str, session_id: str) -> Optional[GeminiBlogGeneratorWithMemory]:
        with self.lock:
            if user_id in self.sessions and session_id in self.sessions[user_id]:
                self.session_last_activity[user_id][session_id] = datetime.now()
                return self.sessions[user_id][session_id]
            return None

    def list_user_sessions(self, user_id: str) -> list[str]:
        with self.lock:
            return list(self.sessions.get(user_id, {}).keys())

    def delete_session(self, user_id: str, session_id: str) -> bool:
        with self.lock:
            if user_id in self.sessions and session_id in self.sessions[user_id]:
                try:
                    self.sessions[user_id][session_id].clear_cache()
                    del self.sessions[user_id][session_id]
                    if user_id in self.session_last_activity and session_id in self.session_last_activity[user_id]:
                        del self.session_last_activity[user_id][session_id]
                    if not self.sessions.get(user_id):
                        self.sessions.pop(user_id, None)
                        self.session_last_activity.pop(user_id, None)
                    memory_file = f"conversations/{user_id}_{session_id}.json"
                    if os.path.exists(memory_file):
                        os.remove(memory_file)
                    return True
                except Exception as e:
                    print(f"Error deleting session: {e}")
                    return False
            return False

    def get_stats(self) -> dict:
        with self.lock:
            total_sessions = sum(len(s) for s in self.sessions.values())
            return {
                "total_users": len(self.sessions),
                "total_sessions": total_sessions,
                "sessions_per_user": {uid: len(s) for uid, s in self.sessions.items()}
            }

# === Global ===
session_manager = SessionManager()

# === Lifespan ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Multi-User Blog Generator API...")
    print(f"💾 Conversations dir: {os.path.abspath('conversations')}")
    yield
    print("🔄 Shutdown: cleaning sessions...")
    for user_id in list(session_manager.sessions.keys()):
        for session_id in list(session_manager.sessions[user_id].keys()):
            session_manager.delete_session(user_id, session_id)
    print("✅ Shutdown complete")

app = FastAPI(
    title="Multi-User Blog Generator API",
    description="FastAPI application with multi-user chat memory for blog generation",
    version="1.0.0",
    lifespan=lifespan
)

# === Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Routes ===

# Serve index.html
@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = Path("templates") / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="index.html not found")
    return FileResponse(index_path)

@app.post("/generate")
async def generate_blog(request: PromptRequest):
    try:
        user_id, session_id, generator = session_manager.get_or_create_session(request.user_id, request.session_id)
        response_text = generator.generate(
            request.user_prompt,
            use_conversation_context=True
        )
        return JSONResponse(content={
            "response": response_text,
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        user_id, session_id, generator = session_manager.get_or_create_session(request.user_id, request.session_id)
        response_text = generator.chat(request.message)
        return JSONResponse(content={
            "response": response_text,
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/stats/{user_id}/{session_id}")
async def get_conversation_stats(user_id: str, session_id: str):
    generator = session_manager.get_session(user_id, session_id)
    if not generator:
        raise HTTPException(status_code=404, detail="Session not found")
    stats = generator.get_conversation_stats()
    return JSONResponse(content={
        "user_id": user_id,
        "session_id": session_id,
        "stats": stats
    })

@app.get("/sessions/{user_id}")
async def list_user_sessions(user_id: str):
    sessions = session_manager.list_user_sessions(user_id)
    return JSONResponse(content={
        "user_id": user_id,
        "sessions": sessions,
        "session_count": len(sessions)
    })

@app.delete("/sessions/{user_id}/{session_id}")
async def delete_user_session(user_id: str, session_id: str):
    success = session_manager.delete_session(user_id, session_id)
    if success:
        return JSONResponse(content={
            "message": "Session deleted successfully",
            "user_id": user_id,
            "session_id": session_id
        })
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.post("/sessions/{user_id}/{session_id}/clear")
async def clear_session_conversation(user_id: str, session_id: str):
    generator = session_manager.get_session(user_id, session_id)
    if not generator:
        raise HTTPException(status_code=404, detail="Session not found")
    generator.clear_conversation()
    return JSONResponse(content={
        "message": "Conversation cleared successfully",
        "user_id": user_id,
        "session_id": session_id
    })

@app.get("/system/stats")
async def get_system_stats():
    stats = session_manager.get_stats()
    return JSONResponse(content={
        "system_stats": stats,
        "timestamp": datetime.now().isoformat()
    })

@app.post("/batch/generate")
async def generate_batch_blogs(request: dict):
    try:
        prompts = request.get("prompts", [])
        user_id = request.get("user_id")
        session_id = request.get("session_id")
        if not prompts:
            raise HTTPException(status_code=400, detail="No prompts provided")
        user_id, session_id, generator = session_manager.get_or_create_session(user_id, session_id)
        results = generator.generate_batch(prompts)
        return JSONResponse(content={
            "results": results,
            "user_id": user_id,
            "session_id": session_id,
            "generated_count": len(results),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    return JSONResponse(content={
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_stats": session_manager.get_stats()
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
