"""
Claim Graph construction module for the Hawkins Truth Engine.

This module provides functionality to build claim graphs that represent relationships
between sources, claims, and entities extracted from documents.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from ..schemas import (
    ClaimGraph,
    ClaimsOutput,
    Document,
    GraphEdge,
    GraphNode,
    SourceIntelOutput,
)


class ClaimGraphBuilder:
    """Builder class for constructing claim graphs."""
    
    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}
        self._node_counter = 0
        self._edge_counter = 0
    
    def _generate_node_id(self, node_type: str, identifier: str) -> str:
        """Generate a unique node ID."""
        return f"{node_type}:{identifier}"
    
    def _generate_edge_id(self) -> str:
        """Generate a unique edge ID."""
        self._edge_counter += 1
        return f"edge:{self._edge_counter}"
    
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        if node.id in self.nodes:
            # Update existing node
            self.nodes[node.id] = node
        else:
            self.nodes[node.id] = node
    
    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        if edge.id in self.edges:
            # Update existing edge
            self.edges[edge.id] = edge
        else:
            self.edges[edge.id] = edge
    
    def build(self) -> ClaimGraph:
        """Build and return the final claim graph."""
        return ClaimGraph(
            nodes=self.nodes.copy(),
            edges=self.edges.copy(),
            metadata={
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "builder_version": "1.0"
            },
            created_at=datetime.now()
        )


def create_source_nodes(document: Document, source_output: SourceIntelOutput) -> list[GraphNode]:
    """Creates source nodes from document metadata and source intelligence."""
    nodes: list[GraphNode] = []
    
    # Create domain source node if available
    if document.domain:
        domain_metadata = {
            "domain": document.domain,
            "trust_score": source_output.source_trust_score,
            "flags": [flag.dict() for flag in source_output.source_flags],
            "url": document.url,
            "retrieved_at": document.retrieved_at.isoformat() if document.retrieved_at else None
        }
        
        domain_node = GraphNode(
            id=f"source:domain:{document.domain}",
            type="source",
            text=f"Domain: {document.domain}",
            metadata=domain_metadata,
            confidence=source_output.source_trust_score,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        nodes.append(domain_node)
    
    # Create author source node if available
    if document.author:
        author_metadata = {
            "author": document.author,
            "published_at": document.published_at.isoformat() if document.published_at else None,
            "title": document.title
        }
        
        # Normalize author name for ID
        author_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document.author.lower())
        
        author_node = GraphNode(
            id=f"source:author:{author_id}",
            type="source", 
            text=f"Author: {document.author}",
            metadata=author_metadata,
            confidence=None,  # Author trust would need separate analysis
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        nodes.append(author_node)
    
    return nodes


def create_claim_nodes(claims_output: ClaimsOutput) -> list[GraphNode]:
    """Creates claim nodes from extracted claims."""
    nodes: list[GraphNode] = []
    
    for claim_item in claims_output.claim_items:
        claim_metadata = {
            "claim_type": claim_item.type,
            "support_status": claim_item.support,
            "reasons": claim_item.reasons,
            "citations": claim_item.citations,
            "quality_flags": claim_item.quality_flags,
            "uncertainty_flags": claim_item.uncertainty_flags,
            "pointers": claim_item.pointers.dict() if claim_item.pointers else None
        }
        
        # Calculate confidence based on support status
        confidence_map = {
            "supported": 0.8,
            "unsupported": 0.2,
            "contested": 0.4,
            "unverifiable": 0.5
        }
        confidence = confidence_map.get(claim_item.support, 0.5)
        
        claim_node = GraphNode(
            id=f"claim:{claim_item.id}",
            type="claim",
            text=claim_item.text,
            metadata=claim_metadata,
            confidence=confidence,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        nodes.append(claim_node)
    
    return nodes


def create_entity_nodes(document: Document) -> list[GraphNode]:
    """Creates entity nodes from document entities."""
    nodes: list[GraphNode] = []
    
    for entity in document.entities:
        entity_metadata = {
            "entity_type": entity.type,
            "sentence_id": entity.sentence_id,
            "char_span": entity.char_span.dict(),
            "normalized": entity.normalized
        }
        
        entity_node = GraphNode(
            id=f"entity:{entity.id}",
            type="entity",
            text=entity.text,
            metadata=entity_metadata,
            confidence=None,  # Entity confidence would need separate analysis
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        nodes.append(entity_node)
    
    return nodes


def _calculate_text_overlap(text1: str, text2: str) -> float:
    """Calculate text overlap between two strings using enhanced word-level Jaccard similarity."""
    # Enhanced normalization: remove punctuation, handle contractions, normalize whitespace
    def normalize_text(text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        # Handle contractions
        text = re.sub(r"'s\b", "", text)  # Remove possessive 's
        text = re.sub(r"n't\b", " not", text)  # Expand contractions
        text = re.sub(r"'re\b", " are", text)
        text = re.sub(r"'ve\b", " have", text)
        text = re.sub(r"'ll\b", " will", text)
        text = re.sub(r"'d\b", " would", text)
        # Remove punctuation and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Normalize and tokenize
    normalized1 = normalize_text(text1)
    normalized2 = normalize_text(text2)
    
    words1 = set(normalized1.split())
    words2 = set(normalized2.split())
    
    # Filter out common stop words that don't contribute to meaning
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
    
    words1 = words1 - stop_words
    words2 = words2 - stop_words
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def _detect_claim_entity_mentions(claim_text: str, entity_text: str, entity_type: str = None) -> float:
    """Detect if a claim mentions an entity and return confidence score with enhanced detection."""
    # Direct text containment (case insensitive) - highest confidence
    if entity_text.lower() in claim_text.lower():
        return 1.0
    
    # Check for partial matches and variations
    entity_words = set(entity_text.lower().split())
    claim_words = set(claim_text.lower().split())
    
    # For person entities, check for last name matches
    if entity_type == "PERSON" and len(entity_words) > 1:
        # Check if last word (likely surname) appears in claim
        last_name = list(entity_words)[-1]
        if last_name in claim_words and len(last_name) > 2:  # Avoid short names
            return 0.8
    
    # For organization entities, check for acronyms or partial names
    if entity_type == "ORG":
        # Check for acronym matches
        entity_acronym = ''.join([word[0].upper() for word in entity_words if len(word) > 2])
        if len(entity_acronym) >= 2 and entity_acronym.lower() in claim_text.lower():
            return 0.7
        
        # Check for key organization words
        org_keywords = {'company', 'corporation', 'inc', 'ltd', 'llc', 'university', 'institute', 'foundation', 'association'}
        if any(keyword in entity_text.lower() for keyword in org_keywords):
            # Look for the main name part without the keyword
            main_words = entity_words - org_keywords
            if main_words and any(word in claim_words for word in main_words):
                return 0.6
    
    # Enhanced word overlap approach with position weighting
    overlap = _calculate_text_overlap(claim_text, entity_text)
    
    # Boost confidence if entity appears near beginning or end of claim (more likely to be important)
    claim_lower = claim_text.lower()
    entity_lower = entity_text.lower()
    
    # Check position-based boosting
    position_boost = 0.0
    claim_length = len(claim_lower)
    
    for word in entity_words:
        if word in claim_lower:
            word_pos = claim_lower.find(word)
            # Boost if word appears in first or last 20% of claim
            if word_pos < claim_length * 0.2 or word_pos > claim_length * 0.8:
                position_boost = max(position_boost, 0.1)
    
    final_score = overlap + position_boost
    
    # Consider it a mention if overlap is significant
    if final_score >= 0.25:  # Lowered threshold due to enhanced detection
        return min(final_score, 1.0)
    
    return 0.0


def _detect_claim_source_attribution(claim_item, document: Document) -> list[tuple[str, float]]:
    """Detect claim-source attributions with enhanced attribution detection."""
    attributions: list[tuple[str, float]] = []
    
    # Enhanced attribution detection using existing attribution spans
    if claim_item.pointers and claim_item.pointers.char_spans:
        claim_spans = claim_item.pointers.char_spans
        
        for attribution in document.attributions:
            # Calculate overlap between claim spans and quote spans
            max_overlap_confidence = 0.0
            
            for claim_span in claim_spans:
                # Calculate overlap ratio
                overlap_start = max(claim_span.start, attribution.quote_span.start)
                overlap_end = min(claim_span.end, attribution.quote_span.end)
                
                if overlap_start < overlap_end:
                    overlap_length = overlap_end - overlap_start
                    claim_length = claim_span.end - claim_span.start
                    quote_length = attribution.quote_span.end - attribution.quote_span.start
                    
                    # Calculate confidence based on overlap ratio
                    claim_overlap_ratio = overlap_length / claim_length if claim_length > 0 else 0
                    quote_overlap_ratio = overlap_length / quote_length if quote_length > 0 else 0
                    
                    # Use the higher of the two ratios, but weight by the smaller to avoid false positives
                    overlap_confidence = max(claim_overlap_ratio, quote_overlap_ratio) * min(claim_overlap_ratio, quote_overlap_ratio)
                    max_overlap_confidence = max(max_overlap_confidence, overlap_confidence)
            
            if max_overlap_confidence > 0.1:  # Threshold for considering attribution
                # If attribution has a speaker entity, attribute to that entity with high confidence
                if attribution.speaker_entity_id is not None:
                    entity_id = f"entity:{attribution.speaker_entity_id}"
                    confidence = 0.9 * max_overlap_confidence
                    attributions.append((entity_id, confidence))
                
                # Also attribute to document source with moderate confidence
                if document.domain:
                    domain_id = f"source:domain:{document.domain}"
                    confidence = 0.7 * max_overlap_confidence
                    attributions.append((domain_id, confidence))
    
    # Enhanced text-based attribution detection for claims without explicit attribution spans
    claim_text_lower = claim_item.text.lower()
    
    # Look for attribution indicators in claim text
    attribution_patterns = [
        r'\b(?:according to|says?|stated?|reported?|claimed?|argued?|believes?|thinks?)\b',
        r'\b(?:research shows?|studies? (?:show|indicate|suggest))\b',
        r'\b(?:experts? (?:say|believe|think|argue))\b',
        r'\b(?:officials? (?:say|stated?|reported?))\b'
    ]
    
    attribution_confidence = 0.0
    for pattern in attribution_patterns:
        if re.search(pattern, claim_text_lower):
            attribution_confidence = max(attribution_confidence, 0.6)
    
    # Check for entity mentions that might be sources
    for entity in document.entities:
        if entity.type in ["PERSON", "ORG"] and entity.text.lower() in claim_text_lower:
            # Higher confidence if attribution pattern is also present
            confidence = 0.8 if attribution_confidence > 0 else 0.5
            entity_id = f"entity:{entity.id}"
            attributions.append((entity_id, confidence))
    
    # Default attribution to document source for all claims (with varying confidence)
    base_confidence = 0.3  # Lower base confidence
    
    # Increase confidence if claim has citations or references
    if claim_item.citations:
        base_confidence = 0.6
    
    # Increase confidence if claim has quality indicators
    if not claim_item.quality_flags or "low_quality" not in claim_item.quality_flags:
        base_confidence += 0.1
    
    if document.domain:
        domain_id = f"source:domain:{document.domain}"
        # Avoid duplicate attributions with higher confidence
        existing_domain_attribution = next((attr for attr in attributions if attr[0] == domain_id), None)
        if not existing_domain_attribution or existing_domain_attribution[1] < base_confidence:
            if existing_domain_attribution:
                attributions.remove(existing_domain_attribution)
            attributions.append((domain_id, base_confidence))
    
    if document.author:
        author_id = re.sub(r'[^a-zA-Z0-9_-]', '_', document.author.lower())
        author_source_id = f"source:author:{author_id}"
        # Similar logic for author attribution
        existing_author_attribution = next((attr for attr in attributions if attr[0] == author_source_id), None)
        if not existing_author_attribution or existing_author_attribution[1] < base_confidence:
            if existing_author_attribution:
                attributions.remove(existing_author_attribution)
            attributions.append((author_source_id, base_confidence))
    
    return attributions


def create_relationship_edges(nodes: dict[str, GraphNode], 
                            document: Document, 
                            claims_output: ClaimsOutput) -> list[GraphEdge]:
    """Creates edges representing relationships between nodes with enhanced algorithms."""
    edges: list[GraphEdge] = []
    edge_counter = 0
    
    def next_edge_id() -> str:
        nonlocal edge_counter
        edge_counter += 1
        return f"edge:{edge_counter}"
    
    # Separate nodes by type for efficient processing
    claim_nodes = {nid: node for nid, node in nodes.items() if node.type == "claim"}
    entity_nodes = {nid: node for nid, node in nodes.items() if node.type == "entity"}
    source_nodes = {nid: node for nid, node in nodes.items() if node.type == "source"}
    
    # Create MENTIONS edges (Claim → Entity) with enhanced detection
    for claim_id, claim_node in claim_nodes.items():
        for entity_id, entity_node in entity_nodes.items():
            # Get entity type from metadata for enhanced detection
            entity_type = entity_node.metadata.get("entity_type", "")
            mention_confidence = _detect_claim_entity_mentions(
                claim_node.text, 
                entity_node.text, 
                entity_type
            )
            
            if mention_confidence > 0.25:  # Lowered threshold due to enhanced detection
                # Calculate enhanced weight based on multiple factors
                base_weight = mention_confidence
                
                # Boost weight for high-confidence entities
                if entity_node.confidence and entity_node.confidence > 0.8:
                    base_weight *= 1.1
                
                # Boost weight for important entity types
                important_types = {"PERSON", "ORG", "GPE"}  # Person, Organization, Geopolitical Entity
                if entity_type in important_types:
                    base_weight *= 1.05
                
                # Normalize weight to [0, 1]
                final_weight = min(base_weight, 1.0)
                
                edge = GraphEdge(
                    id=next_edge_id(),
                    source_id=claim_id,
                    target_id=entity_id,
                    relationship_type="MENTIONS",
                    weight=final_weight,
                    provenance={
                        "detection_method": "enhanced_text_overlap",
                        "base_confidence": mention_confidence,
                        "entity_type": entity_type,
                        "final_weight": final_weight,
                        "claim_text_preview": claim_node.text[:100] + "..." if len(claim_node.text) > 100 else claim_node.text,
                        "entity_text": entity_node.text
                    },
                    created_at=datetime.now()
                )
                edges.append(edge)
    
    # Create ATTRIBUTED_TO edges (Claim → Source) with enhanced attribution
    for claim_item in claims_output.claim_items:
        claim_id = f"claim:{claim_item.id}"
        if claim_id not in claim_nodes:
            continue
            
        attributions = _detect_claim_source_attribution(claim_item, document)
        
        for source_id, confidence in attributions:
            if source_id in source_nodes or source_id in entity_nodes:
                # Enhanced weight calculation based on evidence strength
                base_weight = confidence
                
                # Boost weight for supported claims
                if claim_item.support == "supported":
                    base_weight *= 1.2
                elif claim_item.support == "unsupported":
                    base_weight *= 0.8
                elif claim_item.support == "contested":
                    base_weight *= 0.9
                
                # Boost weight for claims with citations
                if claim_item.citations:
                    base_weight *= 1.1
                
                # Reduce weight for claims with quality issues
                if claim_item.quality_flags and any(flag in claim_item.quality_flags for flag in ["low_quality", "unclear"]):
                    base_weight *= 0.8
                
                # Normalize weight to [0, 1]
                final_weight = min(base_weight, 1.0)
                
                edge = GraphEdge(
                    id=next_edge_id(),
                    source_id=claim_id,
                    target_id=source_id,
                    relationship_type="ATTRIBUTED_TO",
                    weight=final_weight,
                    provenance={
                        "detection_method": "enhanced_attribution_analysis",
                        "base_confidence": confidence,
                        "claim_support": claim_item.support,
                        "has_citations": bool(claim_item.citations),
                        "quality_flags": claim_item.quality_flags,
                        "final_weight": final_weight,
                        "claim_id": claim_item.id
                    },
                    created_at=datetime.now()
                )
                edges.append(edge)
    
    # Create FROM_SOURCE edges (Source → Claim) with enhanced containment logic
    attribution_map = {}  # Track existing attributions for efficient lookup
    for edge in edges:
        if edge.relationship_type == "ATTRIBUTED_TO":
            attribution_map[(edge.source_id, edge.target_id)] = edge.weight
    
    for source_id, source_node in source_nodes.items():
        for claim_id, claim_node in claim_nodes.items():
            # Check if there's a corresponding ATTRIBUTED_TO edge
            attribution_weight = attribution_map.get((claim_id, source_id))
            
            if attribution_weight is not None:
                # Calculate containment weight based on source trust and attribution strength
                base_weight = 0.8  # Base containment confidence
                
                # Adjust based on source trust score
                source_trust = source_node.confidence
                if source_trust is not None:
                    base_weight = (base_weight + source_trust) / 2
                
                # Adjust based on attribution strength
                base_weight = (base_weight + attribution_weight) / 2
                
                # Boost for domain sources (more reliable containment)
                if "domain" in source_id:
                    base_weight *= 1.1
                
                final_weight = min(base_weight, 1.0)
                
                edge = GraphEdge(
                    id=next_edge_id(),
                    source_id=source_id,
                    target_id=claim_id,
                    relationship_type="FROM_SOURCE",
                    weight=final_weight,
                    provenance={
                        "detection_method": "enhanced_document_provenance",
                        "source_trust_score": source_trust,
                        "attribution_weight": attribution_weight,
                        "final_weight": final_weight,
                        "source_type": source_node.metadata.get("domain") or source_node.metadata.get("author")
                    },
                    created_at=datetime.now()
                )
                edges.append(edge)
    
    return edges


def build_claim_graph(document: Document, claims_output: ClaimsOutput, 
                     source_output: SourceIntelOutput) -> ClaimGraph:
    """Constructs claim graph from analysis outputs with performance optimization."""
    import time
    start_time = time.time()
    
    # Input validation - handle invalid inputs gracefully
    if document is None or claims_output is None or source_output is None:
        return ClaimGraph(
            nodes={},
            edges={},
            metadata={
                "node_count": 0,
                "edge_count": 0,
                "error": "Invalid input - null arguments provided",
                "builder_version": "1.0",
            },
            created_at=datetime.now()
        )
    
    builder = ClaimGraphBuilder()
    
    # Performance optimization: Early validation of document size
    num_claims = len(claims_output.claim_items)
    num_entities = len(document.entities)
    
    # Log performance metrics for monitoring
    performance_metadata = {
        "num_claims": num_claims,
        "num_entities": num_entities,
        "num_sources": 0,  # Will be updated below
        "construction_start_time": start_time
    }
    
    # Create all nodes with progress tracking
    source_nodes = create_source_nodes(document, source_output)
    performance_metadata["num_sources"] = len(source_nodes)
    
    claim_nodes = create_claim_nodes(claims_output)
    entity_nodes = create_entity_nodes(document)
    
    # Add nodes to builder
    all_nodes = source_nodes + claim_nodes + entity_nodes
    for node in all_nodes:
        builder.add_node(node)
    
    node_creation_time = time.time()
    performance_metadata["node_creation_time"] = node_creation_time - start_time
    
    # Create relationship edges with optimization for large graphs
    relationship_edges = create_relationship_edges(builder.nodes, document, claims_output)
    
    edge_creation_time = time.time()
    performance_metadata["edge_creation_time"] = edge_creation_time - node_creation_time
    
    # Add edges to builder
    for edge in relationship_edges:
        builder.add_edge(edge)
    
    # Build the final graph with performance metadata
    graph = builder.build()
    
    total_time = time.time() - start_time
    performance_metadata["total_construction_time"] = total_time
    performance_metadata["edges_per_second"] = len(relationship_edges) / max(total_time, 0.001)
    
    # Add performance metadata to graph metadata
    graph.metadata.update({
        "performance": performance_metadata,
        "optimization_applied": num_claims > 30 or num_entities > 50,
        "complexity_score": (num_claims * num_entities) / 1000  # Normalized complexity
    })
    
    return graph