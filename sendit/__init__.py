"""Send It — personalized climbing beta optimizer.

Pipeline: video -> pose (MediaPipe) -> holds -> observed hand sequence
-> climber-specific hold graph -> movement cost -> Dijkstra over
hand-pair states -> optimized beta -> observed vs optimized comparison.
"""
__version__ = "0.1.0"
