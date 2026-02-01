"""
Graph module for the Hawkins Truth Engine.

This module provides graph-based relationship modeling for sources, claims, and entities,
including claim graphs and evidence graphs for enhanced credibility analysis.
"""

from .claim_graph import build_claim_graph, ClaimGraphBuilder
from .evidence_graph import (
    build_evidence_graph,
    calculate_claim_similarity,
    determine_evidence_relationship,
    create_evidence_edges
)

__all__ = [
    "build_claim_graph",
    "ClaimGraphBuilder",
    "build_evidence_graph",
    "calculate_claim_similarity", 
    "determine_evidence_relationship",
    "create_evidence_edges",
]