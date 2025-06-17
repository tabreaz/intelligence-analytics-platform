"""
Metrics logger for recording resource usage metrics to JSON files.
Prevents memory leaks by writing metrics to disk instead of keeping in memory.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio
from collections import deque
import threading

logger = logging.getLogger(__name__)


class MetricsLogger:
    """
    Logs metrics to JSON files with rotation and aggregation support.
    Uses a write buffer to batch writes for better performance.
    """
    
    def __init__(self, 
                 metrics_dir: str = "logs/metrics",
                 buffer_size: int = 100,
                 flush_interval: int = 30):
        """
        Initialize metrics logger.
        
        Args:
            metrics_dir: Directory to store metrics files
            buffer_size: Number of metrics to buffer before writing
            flush_interval: Seconds between automatic flushes
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Separate buffers for different metric types
        self.query_buffer = deque(maxlen=buffer_size)
        self.agent_buffer = deque(maxlen=buffer_size)
        self.error_buffer = deque(maxlen=buffer_size)
        
        # Lock for thread-safe operations
        self._lock = threading.Lock()
        
        # Start time for uptime calculation
        self.start_time = datetime.now()
        
        # Background task for periodic flushing
        self._flush_task = None
        self._running = False
        
    async def start(self):
        """Start the background flush task."""
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info("Metrics logger started")
        
    async def stop(self):
        """Stop the background flush task and flush remaining metrics."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self.flush_all()
        logger.info("Metrics logger stopped")
        
    async def _periodic_flush(self):
        """Periodically flush metrics to disk."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
                
    async def record_query(self, user_id: str, duration: float, success: bool = True,
                     agent_name: Optional[str] = None, query_type: Optional[str] = None):
        """Record query metrics."""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'type': 'query',
            'user_id': user_id,
            'duration': duration,
            'success': success,
            'agent_name': agent_name,
            'query_type': query_type
        }
        
        with self._lock:
            self.query_buffer.append(metric)
            
        # Auto-flush if buffer is full
        if len(self.query_buffer) >= self.buffer_size:
            asyncio.create_task(self._flush_queries())
            
    async def record_agent_execution(self, agent_name: str, duration: float, 
                              success: bool = True, error: Optional[str] = None):
        """Record agent execution metrics."""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'type': 'agent_execution',
            'agent_name': agent_name,
            'duration': duration,
            'success': success,
            'error': error
        }
        
        with self._lock:
            self.agent_buffer.append(metric)
            
        # Auto-flush if buffer is full
        if len(self.agent_buffer) >= self.buffer_size:
            asyncio.create_task(self._flush_agents())
            
    async def record_error(self, error_type: str, error_message: str, 
                     user_id: Optional[str] = None, agent_name: Optional[str] = None):
        """Record error metrics."""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'type': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'user_id': user_id,
            'agent_name': agent_name
        }
        
        with self._lock:
            self.error_buffer.append(metric)
            
        # Auto-flush if buffer is full
        if len(self.error_buffer) >= self.buffer_size:
            asyncio.create_task(self._flush_errors())
            
    async def flush_all(self):
        """Flush all buffers to disk."""
        await asyncio.gather(
            self._flush_queries(),
            self._flush_agents(),
            self._flush_errors(),
            return_exceptions=True
        )
        
    async def _flush_queries(self):
        """Flush query metrics to disk."""
        if not self.query_buffer:
            return
            
        with self._lock:
            metrics = list(self.query_buffer)
            self.query_buffer.clear()
            
        await self._write_metrics('queries', metrics)
        
    async def _flush_agents(self):
        """Flush agent metrics to disk."""
        if not self.agent_buffer:
            return
            
        with self._lock:
            metrics = list(self.agent_buffer)
            self.agent_buffer.clear()
            
        await self._write_metrics('agents', metrics)
        
    async def _flush_errors(self):
        """Flush error metrics to disk."""
        if not self.error_buffer:
            return
            
        with self._lock:
            metrics = list(self.error_buffer)
            self.error_buffer.clear()
            
        await self._write_metrics('errors', metrics)
        
    async def _write_metrics(self, metric_type: str, metrics: list):
        """Write metrics to a JSON file."""
        if not metrics:
            return
            
        # Create filename with date and hour for rotation
        now = datetime.now()
        filename = f"{metric_type}_{now.strftime('%Y%m%d_%H')}.jsonl"
        filepath = self.metrics_dir / filename
        
        try:
            # Append metrics as JSON lines
            with open(filepath, 'a') as f:
                for metric in metrics:
                    f.write(json.dumps(metric) + '\n')
                    
            logger.debug(f"Wrote {len(metrics)} {metric_type} metrics to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to write {metric_type} metrics: {e}")
            
    async def get_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary statistics from recent metrics files.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Summary statistics dictionary
        """
        summary = {
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'query_stats': {'total': 0, 'success': 0, 'failed': 0},
            'agent_stats': {},
            'error_stats': {'total': 0, 'by_type': {}},
            'user_stats': {'unique_users': set()},
            'time_window_hours': hours
        }
        
        # Calculate time window
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        # Read recent metrics files
        for metric_file in self.metrics_dir.glob('*.jsonl'):
            # Skip old files based on filename
            try:
                file_time = datetime.strptime(metric_file.stem.split('_')[1], '%Y%m%d').timestamp()
                if file_time < cutoff_time:
                    continue
            except:
                pass
                
            # Process file
            try:
                with open(metric_file, 'r') as f:
                    for line in f:
                        try:
                            metric = json.loads(line.strip())
                            metric_time = datetime.fromisoformat(metric['timestamp']).timestamp()
                            
                            if metric_time < cutoff_time:
                                continue
                                
                            # Process based on type
                            if metric['type'] == 'query':
                                summary['query_stats']['total'] += 1
                                if metric['success']:
                                    summary['query_stats']['success'] += 1
                                else:
                                    summary['query_stats']['failed'] += 1
                                summary['user_stats']['unique_users'].add(metric.get('user_id'))
                                
                            elif metric['type'] == 'agent_execution':
                                agent = metric['agent_name']
                                if agent not in summary['agent_stats']:
                                    summary['agent_stats'][agent] = {
                                        'executions': 0, 'errors': 0, 
                                        'total_duration': 0.0, 'durations': []
                                    }
                                summary['agent_stats'][agent]['executions'] += 1
                                summary['agent_stats'][agent]['total_duration'] += metric['duration']
                                summary['agent_stats'][agent]['durations'].append(metric['duration'])
                                if not metric['success']:
                                    summary['agent_stats'][agent]['errors'] += 1
                                    
                            elif metric['type'] == 'error':
                                summary['error_stats']['total'] += 1
                                error_type = metric.get('error_type', 'unknown')
                                summary['error_stats']['by_type'][error_type] = \
                                    summary['error_stats']['by_type'].get(error_type, 0) + 1
                                    
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                logger.error(f"Error reading metrics file {metric_file}: {e}")
                
        # Convert sets to counts and calculate averages
        summary['user_stats']['unique_users'] = len(summary['user_stats']['unique_users'])
        
        for agent, stats in summary['agent_stats'].items():
            if stats['durations']:
                stats['avg_duration'] = stats['total_duration'] / stats['executions']
                stats['min_duration'] = min(stats['durations'])
                stats['max_duration'] = max(stats['durations'])
                del stats['durations']  # Remove raw data
                
        return summary
        
    async def cleanup_old_files(self, days_to_keep: int = 7):
        """
        Remove metrics files older than specified days.
        
        Args:
            days_to_keep: Number of days of metrics to keep
        """
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 86400)
        removed_count = 0
        
        for metric_file in self.metrics_dir.glob('*.jsonl'):
            try:
                # Get file modification time
                file_mtime = metric_file.stat().st_mtime
                
                if file_mtime < cutoff_time:
                    metric_file.unlink()
                    removed_count += 1
                    
            except Exception as e:
                logger.error(f"Error removing old metrics file {metric_file}: {e}")
                
        if removed_count > 0:
            logger.info(f"Removed {removed_count} old metrics files")