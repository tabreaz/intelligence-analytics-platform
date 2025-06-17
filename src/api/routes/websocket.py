"""
WebSocket endpoint for real-time activity streaming
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import asyncio
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Store active WebSocket connections per session
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        self.active_connections[session_id].add(websocket)
        logger.info(f"WebSocket connected for session {session_id}")
        
    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.info(f"WebSocket disconnected for session {session_id}")
        
    async def send_activity(self, session_id: str, activity: Dict):
        """Send activity to all connections for a session"""
        if session_id in self.active_connections:
            message = json.dumps({
                "type": "activity",
                "data": {
                    "activity": activity,
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
            
            # Send to all connections for this session
            disconnected = set()
            for websocket in self.active_connections[session_id]:
                try:
                    await websocket.send_text(message)
                except:
                    disconnected.add(websocket)
                    
            # Clean up disconnected sockets
            for ws in disconnected:
                self.active_connections[session_id].discard(ws)
                
    async def send_progress(self, session_id: str, stage: str, progress: int):
        """Send progress update"""
        if session_id in self.active_connections:
            message = json.dumps({
                "type": "progress",
                "data": {
                    "stage": stage,
                    "progress": progress,
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
            
            for websocket in self.active_connections[session_id]:
                try:
                    await websocket.send_text(message)
                except:
                    pass

# Global connection manager
manager = ConnectionManager()

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Keep connection alive and wait for messages
            data = await websocket.receive_text()
            # Could handle client messages here if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, session_id)