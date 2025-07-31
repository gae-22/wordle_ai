"""Performance optimization module for WORDLE ML models - Phase 3 implementation.

This module provides advanced performance optimization techniques including
model compression, caching, profiling, and parallel processing.
"""

import gc
import hashlib
import logging
import pickle
import threading
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from .models import WordleMLModel

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    execution_time: float
    memory_usage: float
    cpu_usage: float
    cache_hits: int = 0
    cache_misses: int = 0
    predictions_per_second: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class PerformanceProfiler:
    """Advanced performance profiler for ML operations."""

    def __init__(self, enable_memory_tracking: bool = True) -> None:
        """Initialize performance profiler.

        Args:
            enable_memory_tracking: Whether to track memory usage
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.metrics_history: dict[str, list[PerformanceMetrics]] = defaultdict(list)
        self._lock = threading.Lock()

        logger.debug("PerformanceProfiler initialized")

    def profile_function(self, name: str):
        """Decorator to profile a function's performance.

        Args:
            name: Name for the profiled operation

        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self._profile_execution(name, func, *args, **kwargs)
            return wrapper
        return decorator

    def _profile_execution(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Profile a function execution.

        Args:
            name: Operation name
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        # Initial measurements
        start_time = time.time()
        process = psutil.Process()

        if self.enable_memory_tracking:
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        initial_cpu = process.cpu_percent()

        try:
            # Execute function
            result = func(*args, **kwargs)

            # Final measurements
            end_time = time.time()
            execution_time = end_time - start_time

            if self.enable_memory_tracking:
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage = final_memory - initial_memory
            else:
                memory_usage = 0.0

            # CPU usage (approximate)
            final_cpu = process.cpu_percent()
            cpu_usage = max(0, final_cpu - initial_cpu)

            # Create metrics
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage
            )

            # Store metrics
            with self._lock:
                self.metrics_history[name].append(metrics)

            logger.debug(f"Profiled {name}: {execution_time:.4f}s, {memory_usage:.2f}MB")

            return result

        except Exception as e:
            logger.error(f"Profiled function {name} failed: {e}")
            raise

    def get_performance_summary(self, operation_name: str | None = None) -> dict[str, Any]:
        """Get performance summary for operations.

        Args:
            operation_name: Specific operation to summarize (None for all)

        Returns:
            Performance summary dictionary
        """
        with self._lock:
            if operation_name:
                if operation_name not in self.metrics_history:
                    return {}
                metrics_to_analyze = {operation_name: self.metrics_history[operation_name]}
            else:
                metrics_to_analyze = dict(self.metrics_history)

        summary = {}

        for name, metrics_list in metrics_to_analyze.items():
            if not metrics_list:
                continue

            execution_times = [m.execution_time for m in metrics_list]
            memory_usages = [m.memory_usage for m in metrics_list]
            cpu_usages = [m.cpu_usage for m in metrics_list]

            summary[name] = {
                'call_count': len(metrics_list),
                'avg_execution_time': np.mean(execution_times),
                'std_execution_time': np.std(execution_times),
                'min_execution_time': np.min(execution_times),
                'max_execution_time': np.max(execution_times),
                'total_execution_time': np.sum(execution_times),
                'avg_memory_usage': np.mean(memory_usages),
                'max_memory_usage': np.max(memory_usages),
                'avg_cpu_usage': np.mean(cpu_usages),
                'performance_trend': self._calculate_trend(execution_times)
            }

        return summary

    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate performance trend.

        Args:
            values: List of performance values

        Returns:
            Trend description
        """
        if len(values) < 5:
            return "insufficient_data"

        # Simple linear trend analysis
        recent = np.mean(values[-5:])
        earlier = np.mean(values[:5])

        if recent < earlier * 0.9:
            return "improving"
        elif recent > earlier * 1.1:
            return "degrading"
        else:
            return "stable"

    def reset_metrics(self, operation_name: str | None = None) -> None:
        """Reset performance metrics.

        Args:
            operation_name: Specific operation to reset (None for all)
        """
        with self._lock:
            if operation_name:
                if operation_name in self.metrics_history:
                    self.metrics_history[operation_name].clear()
            else:
                self.metrics_history.clear()


class SmartCache:
    """Intelligent caching system for ML predictions."""

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: float | None = None,
        enable_persistence: bool = False,
        cache_file: str | None = None
    ) -> None:
        """Initialize smart cache.

        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cache entries (None for no expiration)
            enable_persistence: Whether to persist cache to disk
            cache_file: File to save cache (if persistence enabled)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_persistence = enable_persistence
        self.cache_file = Path(cache_file) if cache_file else Path("cache.pkl")

        self._cache: OrderedDict = OrderedDict()
        self._timestamps: dict[str, float] = {}
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()

        # Load persistent cache
        if self.enable_persistence and self.cache_file.exists():
            self._load_cache()

        logger.debug(f"SmartCache initialized with max_size={max_size}")

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        # Convert arguments to hashable representation
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired.

        Args:
            key: Cache key

        Returns:
            True if expired, False otherwise
        """
        if self.ttl_seconds is None:
            return False

        timestamp = self._timestamps.get(key, 0)
        return time.time() - timestamp > self.ttl_seconds

    def get(self, key: str) -> tuple[Any, bool]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Tuple of (value, hit) where hit indicates cache hit/miss
        """
        with self._lock:
            if key in self._cache and not self._is_expired(key):
                # Cache hit
                value = self._cache[key]
                # Move to end (LRU)
                self._cache.move_to_end(key)
                self._hits += 1
                return value, True
            else:
                # Cache miss
                self._misses += 1
                if key in self._cache:
                    # Remove expired entry
                    del self._cache[key]
                    del self._timestamps[key]
                return None, False

    def put(self, key: str, value: Any) -> None:
        """Put value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Remove oldest entries if cache is full
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]

            # Add new entry
            self._cache[key] = value
            self._timestamps[key] = time.time()

            # Persist if enabled
            if self.enable_persistence:
                self._save_cache()

    def cached_call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with caching.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result (from cache or fresh computation)
        """
        key = self._generate_key(func.__name__, *args, **kwargs)
        value, hit = self.get(key)

        if hit:
            return value
        else:
            # Compute and cache
            result = func(*args, **kwargs)
            self.put(key, result)
            return result

    def cache_decorator(self, func: Callable) -> Callable:
        """Decorator for caching function results.

        Args:
            func: Function to decorate

        Returns:
            Decorated function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.cached_call(func, *args, **kwargs)
        return wrapper

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'expired_entries': sum(1 for key in self._cache if self._is_expired(key))
            }

    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._hits = 0
            self._misses = 0

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            cache_data = {
                'cache': dict(self._cache),
                'timestamps': self._timestamps,
                'hits': self._hits,
                'misses': self._misses
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            self._cache = OrderedDict(cache_data['cache'])
            self._timestamps = cache_data['timestamps']
            self._hits = cache_data.get('hits', 0)
            self._misses = cache_data.get('misses', 0)

            logger.info(f"Loaded cache with {len(self._cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")


class ModelOptimizer:
    """Model-specific optimization techniques."""

    def __init__(self) -> None:
        """Initialize model optimizer."""
        self.optimization_cache = SmartCache(max_size=1000)
        self.profiler = PerformanceProfiler()

        logger.debug("ModelOptimizer initialized")

    @staticmethod
    def quantize_model_weights(model: WordleMLModel, precision: str = "int8") -> WordleMLModel:
        """Quantize model weights to reduce memory usage.

        Args:
            model: Model to quantize
            precision: Target precision ("int8", "int16", "float16")

        Returns:
            Quantized model
        """
        # This is a simplified implementation
        # In practice, would use framework-specific quantization
        logger.info(f"Quantizing model to {precision} precision")

        # For demonstration, we'll just log the operation
        # Real implementation would depend on the model framework
        logger.info("Model quantization completed (simulated)")
        return model

    def optimize_batch_processing(
        self,
        model: WordleMLModel,
        features_list: list[np.ndarray],
        batch_size: int = 32
    ) -> list[np.ndarray]:
        """Optimize predictions using batch processing.

        Args:
            model: Model to use for predictions
            features_list: List of feature arrays
            batch_size: Batch size for processing

        Returns:
            List of predictions
        """
        logger.info(f"Batch processing {len(features_list)} predictions with batch_size={batch_size}")

        all_predictions = []

        # Process in batches
        for i in range(0, len(features_list), batch_size):
            batch = features_list[i:i + batch_size]

            # Stack features for batch prediction
            if batch:
                stacked_features = np.vstack(batch)
                batch_predictions = model.predict(stacked_features)

                # Split predictions back
                current_idx = 0
                for features in batch:
                    pred_size = features.shape[0]
                    pred = batch_predictions[current_idx:current_idx + pred_size]
                    all_predictions.append(pred)
                    current_idx += pred_size

        return all_predictions

    def create_prediction_pipeline(
        self,
        model: WordleMLModel,
        enable_caching: bool = True,
        enable_profiling: bool = True,
        cache_size: int = 5000
    ) -> Callable:
        """Create optimized prediction pipeline.

        Args:
            model: Model to optimize
            enable_caching: Whether to enable caching
            enable_profiling: Whether to enable profiling
            cache_size: Size of prediction cache

        Returns:
            Optimized prediction function
        """
        # Create cache if enabled
        cache = SmartCache(max_size=cache_size) if enable_caching else None

        def optimized_predict(features: np.ndarray) -> np.ndarray:
            """Optimized prediction function."""

            # Profile if enabled
            if enable_profiling:
                with self.profiler.profile_function("prediction"):
                    return self._cached_predict(model, features, cache)
            else:
                return self._cached_predict(model, features, cache)

        return optimized_predict

    def _cached_predict(
        self,
        model: WordleMLModel,
        features: np.ndarray,
        cache: SmartCache | None
    ) -> np.ndarray:
        """Cached prediction with fallback.

        Args:
            model: Model to use
            features: Input features
            cache: Cache object (None to disable caching)

        Returns:
            Predictions
        """
        if cache is None:
            return model.predict(features)

        # Generate cache key
        features_hash = hashlib.md5(features.tobytes()).hexdigest()
        key = f"predict_{features_hash}"

        # Try cache first
        result, hit = cache.get(key)
        if hit:
            return result

        # Compute and cache
        predictions = model.predict(features)
        cache.put(key, predictions)

        return predictions


class ParallelProcessor:
    """Parallel processing utilities for ML operations."""

    def __init__(self, max_workers: int | None = None) -> None:
        """Initialize parallel processor.

        Args:
            max_workers: Maximum number of workers (None for auto-detect)
        """
        self.max_workers = max_workers or min(8, psutil.cpu_count())
        logger.debug(f"ParallelProcessor initialized with {self.max_workers} workers")

    def parallel_predict(
        self,
        models: list[WordleMLModel],
        features: np.ndarray,
        use_processes: bool = False
    ) -> list[np.ndarray]:
        """Run predictions in parallel across multiple models.

        Args:
            models: List of models to use
            features: Input features
            use_processes: Whether to use processes instead of threads

        Returns:
            List of predictions from each model
        """
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        with executor_class(max_workers=min(self.max_workers, len(models))) as executor:
            futures = [
                executor.submit(model.predict, features)
                for model in models
            ]

            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Parallel prediction failed: {e}")
                    results.append(None)

        return results

    def parallel_feature_extraction(
        self,
        feature_extractor: Any,
        data_chunks: list[Any],
        extract_func_name: str
    ) -> list[np.ndarray]:
        """Extract features in parallel.

        Args:
            feature_extractor: Feature extractor object
            data_chunks: List of data chunks to process
            extract_func_name: Name of extraction method

        Returns:
            List of extracted features
        """
        def extract_chunk(chunk):
            extract_func = getattr(feature_extractor, extract_func_name)
            return extract_func(chunk)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(extract_chunk, chunk) for chunk in data_chunks]

            results = []
            for future in futures:
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Parallel feature extraction failed: {e}")
                    results.append(None)

        return [r for r in results if r is not None]


class MemoryOptimizer:
    """Memory usage optimization utilities."""

    def __init__(self) -> None:
        """Initialize memory optimizer."""
        self.memory_tracker = defaultdict(float)
        logger.debug("MemoryOptimizer initialized")

    def optimize_numpy_arrays(self, arrays: list[np.ndarray]) -> list[np.ndarray]:
        """Optimize numpy arrays for memory efficiency.

        Args:
            arrays: List of numpy arrays to optimize

        Returns:
            List of optimized arrays
        """
        optimized = []

        for arr in arrays:
            # Find optimal dtype
            if arr.dtype == np.float64 and np.allclose(arr, arr.astype(np.float32)):
                # Try float32 if precision loss is acceptable
                arr = arr.astype(np.float32)
                logger.debug("Converted float64 to float32")

            # Make read-only if possible (saves memory in copies)
            arr.flags.writeable = False
            optimized.append(arr)

        return optimized

    def cleanup_memory(self) -> dict[str, float]:
        """Force garbage collection and return memory stats.

        Returns:
            Memory statistics before and after cleanup
        """
        process = psutil.Process()

        # Memory before cleanup
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Force cleanup
        gc.collect()

        # Memory after cleanup
        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        memory_freed = memory_before - memory_after

        logger.info(f"Memory cleanup: freed {memory_freed:.2f}MB")

        return {
            'before_mb': memory_before,
            'after_mb': memory_after,
            'freed_mb': memory_freed
        }

    def monitor_memory_usage(self, func: Callable) -> Callable:
        """Decorator to monitor memory usage of a function.

        Args:
            func: Function to monitor

        Returns:
            Decorated function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()

            # Memory before
            memory_before = process.memory_info().rss / 1024 / 1024

            try:
                result = func(*args, **kwargs)

                # Memory after
                memory_after = process.memory_info().rss / 1024 / 1024
                memory_used = memory_after - memory_before

                # Track usage
                func_name = func.__name__
                self.memory_tracker[func_name] = max(
                    self.memory_tracker[func_name],
                    memory_used
                )

                if memory_used > 10:  # Log if > 10MB
                    logger.info(f"{func_name} used {memory_used:.2f}MB")

                return result

            except Exception as e:
                logger.error(f"Memory monitoring failed for {func.__name__}: {e}")
                raise

        return wrapper

    def get_memory_report(self) -> dict[str, Any]:
        """Get memory usage report.

        Returns:
            Dictionary with memory statistics
        """
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'current_memory_mb': memory_info.rss / 1024 / 1024,
            'peak_memory_mb': memory_info.peak_wset / 1024 / 1024 if hasattr(memory_info, 'peak_wset') else 0,
            'function_memory_usage': dict(self.memory_tracker),
            'available_memory_mb': psutil.virtual_memory().available / 1024 / 1024,
            'memory_percent': process.memory_percent()
        }


class PerformanceOptimizationSuite:
    """Complete performance optimization suite."""

    def __init__(
        self,
        enable_caching: bool = True,
        enable_profiling: bool = True,
        enable_parallel: bool = True,
        cache_size: int = 10000
    ) -> None:
        """Initialize optimization suite.

        Args:
            enable_caching: Whether to enable caching
            enable_profiling: Whether to enable profiling
            enable_parallel: Whether to enable parallel processing
            cache_size: Size of caches
        """
        self.enable_caching = enable_caching
        self.enable_profiling = enable_profiling
        self.enable_parallel = enable_parallel

        # Initialize components
        self.cache = SmartCache(max_size=cache_size) if enable_caching else None
        self.profiler = PerformanceProfiler() if enable_profiling else None
        self.parallel_processor = ParallelProcessor() if enable_parallel else None
        self.model_optimizer = ModelOptimizer()
        self.memory_optimizer = MemoryOptimizer()

        logger.info("PerformanceOptimizationSuite initialized")

    def optimize_model(
        self,
        model: WordleMLModel,
        optimization_level: str = "balanced"
    ) -> WordleMLModel:
        """Apply comprehensive optimizations to a model.

        Args:
            model: Model to optimize
            optimization_level: Optimization level ("fast", "balanced", "memory")

        Returns:
            Optimized model
        """
        logger.info(f"Optimizing model with level: {optimization_level}")

        optimized_model = model

        if optimization_level in ["balanced", "memory"]:
            # Apply quantization for memory optimization
            optimized_model = self.model_optimizer.quantize_model_weights(
                optimized_model, precision="float16"
            )

        if optimization_level == "memory":
            # Aggressive memory optimization
            self.memory_optimizer.cleanup_memory()

        return optimized_model

    def create_optimized_predictor(
        self,
        model: WordleMLModel,
        optimization_level: str = "balanced"
    ) -> Callable:
        """Create fully optimized prediction function.

        Args:
            model: Model to create predictor for
            optimization_level: Level of optimization

        Returns:
            Optimized prediction function
        """
        # Optimize model first
        optimized_model = self.optimize_model(model, optimization_level)

        # Create optimized pipeline
        predictor = self.model_optimizer.create_prediction_pipeline(
            optimized_model,
            enable_caching=self.enable_caching,
            enable_profiling=self.enable_profiling
        )

        # Add memory monitoring
        if optimization_level in ["balanced", "memory"]:
            predictor = self.memory_optimizer.monitor_memory_usage(predictor)

        return predictor

    def get_comprehensive_report(self) -> dict[str, Any]:
        """Get comprehensive performance report.

        Returns:
            Dictionary with all performance metrics
        """
        report = {
            'timestamp': time.time(),
            'optimization_status': {
                'caching_enabled': self.enable_caching,
                'profiling_enabled': self.enable_profiling,
                'parallel_enabled': self.enable_parallel
            }
        }

        # Cache statistics
        if self.cache:
            report['cache_stats'] = self.cache.get_stats()

        # Profiling statistics
        if self.profiler:
            report['profiling_stats'] = self.profiler.get_performance_summary()

        # Memory statistics
        report['memory_stats'] = self.memory_optimizer.get_memory_report()

        return report

    def reset_all_metrics(self) -> None:
        """Reset all performance metrics."""
        if self.cache:
            self.cache.clear()
        if self.profiler:
            self.profiler.reset_metrics()
        self.memory_optimizer.memory_tracker.clear()

        logger.info("All performance metrics reset")
