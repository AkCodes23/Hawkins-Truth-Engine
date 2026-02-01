"""
Graph Query Interface module for the Hawkins Truth Engine.

This module provides comprehensive query capabilities for both ClaimGraph and EvidenceGraph
structures, including node queries, relationship queries, graph metrics, and export functionality.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from typing import Any, Literal

from ..schemas import ClaimGraph, EvidenceGraph, GraphEdge, GraphNode


class GraphQueryInterface:
    """Provides query methods for graph structures."""
    
    def __init__(self):
        """Initialize the graph query interface."""
        pass
    
    def find_claims_by_source(self, graph: ClaimGraph, source_id: str) -> list[GraphNode]:
        """
        Returns all claims attributed to a source.
        
        Args:
            graph: The claim graph to query
            source_id: ID of the source node to find claims for
            
        Returns:
            List of claim nodes attributed to the source
        """
        if source_id not in graph.nodes:
            return []
        
        source_node = graph.nodes[source_id]
        if source_node.type != "source":
            return []
        
        claim_nodes = []
        
        # Find claims through FROM_SOURCE edges (Source → Claim)
        for edge in graph.edges.values():
            if (edge.relationship_type == "FROM_SOURCE" and 
                edge.source_id == source_id and 
                edge.target_id in graph.nodes):
                
                target_node = graph.nodes[edge.target_id]
                if target_node.type == "claim":
                    claim_nodes.append(target_node)
        
        # Also find claims through ATTRIBUTED_TO edges (Claim → Source)
        for edge in graph.edges.values():
            if (edge.relationship_type == "ATTRIBUTED_TO" and 
                edge.target_id == source_id and 
                edge.source_id in graph.nodes):
                
                source_node = graph.nodes[edge.source_id]
                if source_node.type == "claim":
                    claim_nodes.append(source_node)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_claims = []
        for claim in claim_nodes:
            if claim.id not in seen:
                seen.add(claim.id)
                unique_claims.append(claim)
        
        return unique_claims
    
    def find_entities_in_claim(self, graph: ClaimGraph, claim_id: str) -> list[GraphNode]:
        """
        Returns all entities mentioned in a claim.
        
        Args:
            graph: The claim graph to query
            claim_id: ID of the claim node to find entities for
            
        Returns:
            List of entity nodes mentioned in the claim
        """
        if claim_id not in graph.nodes:
            return []
        
        claim_node = graph.nodes[claim_id]
        if claim_node.type != "claim":
            return []
        
        entity_nodes = []
        
        # Find entities through MENTIONS edges (Claim → Entity)
        for edge in graph.edges.values():
            if (edge.relationship_type == "MENTIONS" and 
                edge.source_id == claim_id and 
                edge.target_id in graph.nodes):
                
                target_node = graph.nodes[edge.target_id]
                if target_node.type == "entity":
                    entity_nodes.append(target_node)
        
        # Sort by edge weight (confidence) in descending order
        entity_weights = {}
        for edge in graph.edges.values():
            if (edge.relationship_type == "MENTIONS" and 
                edge.source_id == claim_id and 
                edge.target_id in graph.nodes):
                entity_weights[edge.target_id] = edge.weight
        
        entity_nodes.sort(key=lambda node: entity_weights.get(node.id, 0.0), reverse=True)
        
        return entity_nodes
    
    def find_supporting_claims(self, evidence_graph: EvidenceGraph, claim_id: str) -> list[GraphNode]:
        """
        Returns claims that support the given claim.
        
        Args:
            evidence_graph: The evidence graph to query
            claim_id: ID of the claim to find supporting claims for
            
        Returns:
            List of claim nodes that support the given claim
        """
        claim_node_id = evidence_graph.claim_nodes.get(claim_id)
        if not claim_node_id:
            return []
        
        supporting_claims = []
        
        # Find SUPPORTS edges where the target is our claim
        for edge in evidence_graph.edges.values():
            if (edge.relationship_type == "SUPPORTS" and 
                edge.target_id == claim_node_id):
                
                # Extract claim ID from source node ID (format: "claim:claim_id")
                source_claim_id = edge.source_id.split(":", 1)[1] if ":" in edge.source_id else edge.source_id
                
                # Create a GraphNode representation for the supporting claim
                supporting_claim = GraphNode(
                    id=edge.source_id,
                    type="claim",
                    text=f"Supporting claim {source_claim_id}",
                    metadata={
                        "claim_id": source_claim_id,
                        "support_weight": edge.weight,
                        "relationship_provenance": edge.provenance
                    },
                    confidence=edge.weight
                )
                supporting_claims.append(supporting_claim)
        
        # Also find SUPPORTS edges where the source is our claim (bidirectional)
        for edge in evidence_graph.edges.values():
            if (edge.relationship_type == "SUPPORTS" and 
                edge.source_id == claim_node_id):
                
                # Extract claim ID from target node ID
                target_claim_id = edge.target_id.split(":", 1)[1] if ":" in edge.target_id else edge.target_id
                
                supporting_claim = GraphNode(
                    id=edge.target_id,
                    type="claim", 
                    text=f"Supporting claim {target_claim_id}",
                    metadata={
                        "claim_id": target_claim_id,
                        "support_weight": edge.weight,
                        "relationship_provenance": edge.provenance
                    },
                    confidence=edge.weight
                )
                supporting_claims.append(supporting_claim)
        
        # Remove duplicates and sort by support weight
        seen = set()
        unique_supporting = []
        for claim in supporting_claims:
            if claim.id not in seen:
                seen.add(claim.id)
                unique_supporting.append(claim)
        
        unique_supporting.sort(key=lambda x: x.confidence or 0.0, reverse=True)
        
        return unique_supporting
    
    def find_contradicting_claims(self, evidence_graph: EvidenceGraph, claim_id: str) -> list[GraphNode]:
        """
        Returns claims that contradict the given claim.
        
        Args:
            evidence_graph: The evidence graph to query
            claim_id: ID of the claim to find contradicting claims for
            
        Returns:
            List of claim nodes that contradict the given claim
        """
        claim_node_id = evidence_graph.claim_nodes.get(claim_id)
        if not claim_node_id:
            return []
        
        contradicting_claims = []
        
        # Find CONTRADICTS edges where the target is our claim
        for edge in evidence_graph.edges.values():
            if (edge.relationship_type == "CONTRADICTS" and 
                edge.target_id == claim_node_id):
                
                # Extract claim ID from source node ID
                source_claim_id = edge.source_id.split(":", 1)[1] if ":" in edge.source_id else edge.source_id
                
                contradicting_claim = GraphNode(
                    id=edge.source_id,
                    type="claim",
                    text=f"Contradicting claim {source_claim_id}",
                    metadata={
                        "claim_id": source_claim_id,
                        "contradiction_weight": edge.weight,
                        "relationship_provenance": edge.provenance
                    },
                    confidence=edge.weight
                )
                contradicting_claims.append(contradicting_claim)
        
        # Also find CONTRADICTS edges where the source is our claim (bidirectional)
        for edge in evidence_graph.edges.values():
            if (edge.relationship_type == "CONTRADICTS" and 
                edge.source_id == claim_node_id):
                
                # Extract claim ID from target node ID
                target_claim_id = edge.target_id.split(":", 1)[1] if ":" in edge.target_id else edge.target_id
                
                contradicting_claim = GraphNode(
                    id=edge.target_id,
                    type="claim",
                    text=f"Contradicting claim {target_claim_id}",
                    metadata={
                        "claim_id": target_claim_id,
                        "contradiction_weight": edge.weight,
                        "relationship_provenance": edge.provenance
                    },
                    confidence=edge.weight
                )
                contradicting_claims.append(contradicting_claim)
        
        # Remove duplicates and sort by contradiction weight
        seen = set()
        unique_contradicting = []
        for claim in contradicting_claims:
            if claim.id not in seen:
                seen.add(claim.id)
                unique_contradicting.append(claim)
        
        unique_contradicting.sort(key=lambda x: x.confidence or 0.0, reverse=True)
        
        return unique_contradicting
    
    def find_related_claims(self, evidence_graph: EvidenceGraph, claim_id: str) -> list[GraphNode]:
        """
        Returns claims that are topically related to the given claim.
        
        Args:
            evidence_graph: The evidence graph to query
            claim_id: ID of the claim to find related claims for
            
        Returns:
            List of claim nodes that are topically related to the given claim
        """
        claim_node_id = evidence_graph.claim_nodes.get(claim_id)
        if not claim_node_id:
            return []
        
        related_claims = []
        
        # Find RELATES_TO edges where the target is our claim
        for edge in evidence_graph.edges.values():
            if (edge.relationship_type == "RELATES_TO" and 
                edge.target_id == claim_node_id):
                
                source_claim_id = edge.source_id.split(":", 1)[1] if ":" in edge.source_id else edge.source_id
                
                related_claim = GraphNode(
                    id=edge.source_id,
                    type="claim",
                    text=f"Related claim {source_claim_id}",
                    metadata={
                        "claim_id": source_claim_id,
                        "relation_weight": edge.weight,
                        "relationship_provenance": edge.provenance
                    },
                    confidence=edge.weight
                )
                related_claims.append(related_claim)
        
        # Also find RELATES_TO edges where the source is our claim (bidirectional)
        for edge in evidence_graph.edges.values():
            if (edge.relationship_type == "RELATES_TO" and 
                edge.source_id == claim_node_id):
                
                target_claim_id = edge.target_id.split(":", 1)[1] if ":" in edge.target_id else edge.target_id
                
                related_claim = GraphNode(
                    id=edge.target_id,
                    type="claim",
                    text=f"Related claim {target_claim_id}",
                    metadata={
                        "claim_id": target_claim_id,
                        "relation_weight": edge.weight,
                        "relationship_provenance": edge.provenance
                    },
                    confidence=edge.weight
                )
                related_claims.append(related_claim)
        
        # Remove duplicates and sort by relation weight
        seen = set()
        unique_related = []
        for claim in related_claims:
            if claim.id not in seen:
                seen.add(claim.id)
                unique_related.append(claim)
        
        unique_related.sort(key=lambda x: x.confidence or 0.0, reverse=True)
        
        return unique_related
    
    def calculate_node_centrality(self, graph: ClaimGraph | EvidenceGraph) -> dict[str, float]:
        """
        Calculates centrality metrics for nodes.
        
        Args:
            graph: The graph to calculate centrality for
            
        Returns:
            Dictionary mapping node IDs to centrality scores
        """
        if isinstance(graph, ClaimGraph):
            nodes = graph.nodes
            edges = graph.edges
        else:  # EvidenceGraph
            # For evidence graphs, create virtual nodes from claim_nodes
            nodes = {node_id: GraphNode(
                id=node_id,
                type="claim",
                text=f"Claim {claim_id}",
                metadata={"claim_id": claim_id}
            ) for claim_id, node_id in graph.claim_nodes.items()}
            edges = graph.edges
        
        if not nodes:
            return {}
        
        # Calculate degree centrality (number of connections)
        degree_centrality = defaultdict(int)
        
        for edge in edges.values():
            if edge.source_id in nodes:
                degree_centrality[edge.source_id] += 1
            if edge.target_id in nodes:
                degree_centrality[edge.target_id] += 1
        
        # Normalize degree centrality
        max_possible_degree = len(nodes) - 1
        if max_possible_degree > 0:
            for node_id in degree_centrality:
                degree_centrality[node_id] = degree_centrality[node_id] / max_possible_degree
        
        # Calculate weighted centrality (sum of edge weights)
        weighted_centrality = defaultdict(float)
        
        for edge in edges.values():
            weight = edge.weight
            if edge.source_id in nodes:
                weighted_centrality[edge.source_id] += weight
            if edge.target_id in nodes:
                weighted_centrality[edge.target_id] += weight
        
        # Normalize weighted centrality
        max_weighted = max(weighted_centrality.values()) if weighted_centrality else 1.0
        if max_weighted > 0:
            for node_id in weighted_centrality:
                weighted_centrality[node_id] = weighted_centrality[node_id] / max_weighted
        
        # Calculate betweenness centrality (simplified version)
        betweenness_centrality = self._calculate_betweenness_centrality(nodes, edges)
        
        # Combine centrality measures
        centrality_scores = {}
        for node_id in nodes:
            degree_score = degree_centrality.get(node_id, 0.0)
            weighted_score = weighted_centrality.get(node_id, 0.0)
            betweenness_score = betweenness_centrality.get(node_id, 0.0)
            
            # Weighted combination of centrality measures
            combined_score = (
                degree_score * 0.4 +
                weighted_score * 0.4 +
                betweenness_score * 0.2
            )
            
            centrality_scores[node_id] = combined_score
        
        return centrality_scores
    
    def _calculate_betweenness_centrality(self, nodes: dict[str, GraphNode], 
                                        edges: dict[str, GraphEdge]) -> dict[str, float]:
        """
        Calculate betweenness centrality using a simplified algorithm.
        
        Args:
            nodes: Dictionary of nodes
            edges: Dictionary of edges
            
        Returns:
            Dictionary mapping node IDs to betweenness centrality scores
        """
        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in edges.values():
            adjacency[edge.source_id].append(edge.target_id)
            adjacency[edge.target_id].append(edge.source_id)  # Treat as undirected
        
        betweenness = defaultdict(float)
        node_list = list(nodes.keys())
        
        # For each pair of nodes, find shortest paths and count how many pass through each node
        for i, source in enumerate(node_list):
            for j, target in enumerate(node_list):
                if i >= j:  # Avoid duplicate pairs and self-pairs
                    continue
                
                # Find shortest path using BFS
                paths = self._find_shortest_paths(source, target, adjacency)
                
                if paths:
                    # Count nodes that appear in shortest paths
                    for path in paths:
                        for node in path[1:-1]:  # Exclude source and target
                            betweenness[node] += 1.0 / len(paths)
        
        # Normalize betweenness centrality
        n = len(nodes)
        if n > 2:
            normalization_factor = (n - 1) * (n - 2) / 2
            for node_id in betweenness:
                betweenness[node_id] = betweenness[node_id] / normalization_factor
        
        return dict(betweenness)
    
    def _find_shortest_paths(self, source: str, target: str, 
                           adjacency: dict[str, list[str]]) -> list[list[str]]:
        """
        Find all shortest paths between source and target nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            adjacency: Adjacency list representation of the graph
            
        Returns:
            List of shortest paths (each path is a list of node IDs)
        """
        if source == target:
            return [[source]]
        
        # BFS to find shortest paths
        queue = deque([(source, [source])])
        visited = {source: 0}
        paths = []
        shortest_distance = None
        
        while queue:
            current_node, path = queue.popleft()
            current_distance = len(path) - 1
            
            # If we've found a longer path than the shortest, stop
            if shortest_distance is not None and current_distance > shortest_distance:
                break
            
            for neighbor in adjacency.get(current_node, []):
                new_path = path + [neighbor]
                new_distance = len(new_path) - 1
                
                if neighbor == target:
                    if shortest_distance is None:
                        shortest_distance = new_distance
                    if new_distance == shortest_distance:
                        paths.append(new_path)
                elif neighbor not in visited or visited[neighbor] == new_distance:
                    visited[neighbor] = new_distance
                    queue.append((neighbor, new_path))
        
        return paths
    
    def calculate_graph_density(self, graph: ClaimGraph | EvidenceGraph) -> float:
        """
        Calculate the density of the graph (ratio of actual edges to possible edges).
        
        Args:
            graph: The graph to calculate density for
            
        Returns:
            Graph density score between 0.0 and 1.0
        """
        if isinstance(graph, ClaimGraph):
            num_nodes = len(graph.nodes)
            num_edges = len(graph.edges)
        else:  # EvidenceGraph
            num_nodes = len(graph.claim_nodes)
            num_edges = len(graph.edges)
        
        if num_nodes < 2:
            return 0.0
        
        # For directed graphs, maximum possible edges is n * (n - 1)
        # For undirected graphs, it's n * (n - 1) / 2
        # We'll treat our graphs as directed for this calculation
        max_possible_edges = num_nodes * (num_nodes - 1)
        
        if max_possible_edges == 0:
            return 0.0
        
        return num_edges / max_possible_edges
    
    def get_subgraph(self, graph: ClaimGraph, node_filter: callable = None, 
                    edge_filter: callable = None) -> ClaimGraph:
        """
        Extract a subgraph based on node and edge filters.
        
        Args:
            graph: The source graph
            node_filter: Function that takes a GraphNode and returns True to include it
            edge_filter: Function that takes a GraphEdge and returns True to include it
            
        Returns:
            New ClaimGraph containing only filtered nodes and edges
        """
        # Filter nodes
        filtered_nodes = {}
        if node_filter:
            for node_id, node in graph.nodes.items():
                if node_filter(node):
                    filtered_nodes[node_id] = node
        else:
            filtered_nodes = graph.nodes.copy()
        
        # Filter edges (only include edges where both nodes are in filtered set)
        filtered_edges = {}
        for edge_id, edge in graph.edges.items():
            if (edge.source_id in filtered_nodes and 
                edge.target_id in filtered_nodes):
                
                if edge_filter is None or edge_filter(edge):
                    filtered_edges[edge_id] = edge
        
        # Create subgraph
        subgraph = ClaimGraph(
            nodes=filtered_nodes,
            edges=filtered_edges,
            metadata={
                **graph.metadata,
                "subgraph_info": {
                    "original_nodes": len(graph.nodes),
                    "original_edges": len(graph.edges),
                    "filtered_nodes": len(filtered_nodes),
                    "filtered_edges": len(filtered_edges),
                    "node_filter_applied": node_filter is not None,
                    "edge_filter_applied": edge_filter is not None
                }
            },
            created_at=graph.created_at
        )
        
        return subgraph
    
    def traverse_graph(self, graph: ClaimGraph | EvidenceGraph, start_node_id: str, 
                      max_depth: int = 3, relationship_types: list[str] = None) -> dict[str, Any]:
        """
        Traverse the graph starting from a given node.
        
        Args:
            graph: The graph to traverse
            start_node_id: ID of the starting node
            max_depth: Maximum depth to traverse
            relationship_types: List of relationship types to follow (None for all)
            
        Returns:
            Dictionary containing traversal results with paths and discovered nodes
        """
        if isinstance(graph, ClaimGraph):
            nodes = graph.nodes
            edges = graph.edges
        else:  # EvidenceGraph
            # Create virtual nodes for evidence graph
            nodes = {node_id: GraphNode(
                id=node_id,
                type="claim",
                text=f"Claim {claim_id}",
                metadata={"claim_id": claim_id}
            ) for claim_id, node_id in graph.claim_nodes.items()}
            edges = graph.edges
        
        if start_node_id not in nodes:
            return {
                "error": f"Start node '{start_node_id}' not found in graph",
                "paths": [],
                "discovered_nodes": [],
                "traversal_stats": {}
            }
        
        # Build adjacency list with relationship types
        adjacency = defaultdict(list)
        for edge in edges.values():
            if relationship_types is None or edge.relationship_type in relationship_types:
                adjacency[edge.source_id].append({
                    "target": edge.target_id,
                    "relationship": edge.relationship_type,
                    "weight": edge.weight,
                    "edge_id": edge.id
                })
                # For undirected traversal, add reverse direction
                adjacency[edge.target_id].append({
                    "target": edge.source_id,
                    "relationship": edge.relationship_type,
                    "weight": edge.weight,
                    "edge_id": edge.id
                })
        
        # BFS traversal with path tracking
        queue = deque([(start_node_id, [start_node_id], 0)])  # (node, path, depth)
        visited = {start_node_id}
        paths = []
        discovered_nodes = [nodes[start_node_id]]
        relationship_counts = defaultdict(int)
        
        while queue:
            current_node, path, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            for neighbor_info in adjacency.get(current_node, []):
                neighbor_id = neighbor_info["target"]
                relationship = neighbor_info["relationship"]
                weight = neighbor_info["weight"]
                
                relationship_counts[relationship] += 1
                
                new_path = path + [neighbor_id]
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    discovered_nodes.append(nodes[neighbor_id])
                    queue.append((neighbor_id, new_path, depth + 1))
                
                # Record path with relationship info
                paths.append({
                    "path": new_path,
                    "depth": depth + 1,
                    "relationship": relationship,
                    "weight": weight,
                    "edge_id": neighbor_info["edge_id"]
                })
        
        return {
            "start_node": nodes[start_node_id],
            "paths": paths,
            "discovered_nodes": discovered_nodes,
            "traversal_stats": {
                "total_nodes_discovered": len(discovered_nodes),
                "total_paths": len(paths),
                "max_depth_reached": max([p["depth"] for p in paths]) if paths else 0,
                "relationship_distribution": dict(relationship_counts),
                "average_path_weight": sum(p["weight"] for p in paths) / len(paths) if paths else 0.0
            }
        }
    
    def export_graph(self, graph: ClaimGraph | EvidenceGraph, 
                    format: Literal["json", "graphml", "dot"]) -> str:
        """
        Exports graph in specified format.
        
        Args:
            graph: The graph to export
            format: Export format ("json", "graphml", or "dot")
            
        Returns:
            String representation of the graph in the specified format
        """
        if format == "json":
            return self._export_json(graph)
        elif format == "graphml":
            return self._export_graphml(graph)
        elif format == "dot":
            return self._export_dot(graph)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, graph: ClaimGraph | EvidenceGraph) -> str:
        """Export graph as JSON."""
        if isinstance(graph, ClaimGraph):
            export_data = {
                "type": "ClaimGraph",
                "nodes": [node.dict() for node in graph.nodes.values()],
                "edges": [edge.dict() for edge in graph.edges.values()],
                "metadata": graph.metadata,
                "created_at": graph.created_at.isoformat()
            }
        else:  # EvidenceGraph
            export_data = {
                "type": "EvidenceGraph",
                "claim_nodes": graph.claim_nodes,
                "edges": [edge.dict() for edge in graph.edges.values()],
                "similarity_threshold": graph.similarity_threshold,
                "metadata": graph.metadata,
                "created_at": graph.created_at.isoformat()
            }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def _export_graphml(self, graph: ClaimGraph | EvidenceGraph) -> str:
        """Export graph as GraphML format."""
        graphml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
            '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns',
            '         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">',
            '',
            '  <!-- Node attributes -->',
            '  <key id="node_type" for="node" attr.name="type" attr.type="string"/>',
            '  <key id="node_text" for="node" attr.name="text" attr.type="string"/>',
            '  <key id="node_confidence" for="node" attr.name="confidence" attr.type="double"/>',
            '',
            '  <!-- Edge attributes -->',
            '  <key id="edge_relationship" for="edge" attr.name="relationship_type" attr.type="string"/>',
            '  <key id="edge_weight" for="edge" attr.name="weight" attr.type="double"/>',
            '',
            '  <graph id="G" edgedefault="directed">'
        ]
        
        # Add nodes
        if isinstance(graph, ClaimGraph):
            nodes = graph.nodes
            edges = graph.edges
        else:  # EvidenceGraph
            # Create virtual nodes for evidence graph
            nodes = {node_id: GraphNode(
                id=node_id,
                type="claim",
                text=f"Claim {claim_id}",
                metadata={"claim_id": claim_id}
            ) for claim_id, node_id in graph.claim_nodes.items()}
            edges = graph.edges
        
        for node in nodes.values():
            graphml_lines.extend([
                f'    <node id="{self._escape_xml(node.id)}">',
                f'      <data key="node_type">{self._escape_xml(node.type)}</data>',
                f'      <data key="node_text">{self._escape_xml(node.text)}</data>',
                f'      <data key="node_confidence">{node.confidence or 0.0}</data>',
                '    </node>'
            ])
        
        # Add edges
        for edge in edges.values():
            graphml_lines.extend([
                f'    <edge source="{self._escape_xml(edge.source_id)}" target="{self._escape_xml(edge.target_id)}">',
                f'      <data key="edge_relationship">{self._escape_xml(edge.relationship_type)}</data>',
                f'      <data key="edge_weight">{edge.weight}</data>',
                '    </edge>'
            ])
        
        graphml_lines.extend([
            '  </graph>',
            '</graphml>'
        ])
        
        return '\n'.join(graphml_lines)
    
    def _export_dot(self, graph: ClaimGraph | EvidenceGraph) -> str:
        """Export graph as DOT format for Graphviz."""
        dot_lines = [
            'digraph G {',
            '  rankdir=LR;',
            '  node [shape=box, style=rounded];',
            ''
        ]
        
        if isinstance(graph, ClaimGraph):
            nodes = graph.nodes
            edges = graph.edges
        else:  # EvidenceGraph
            # Create virtual nodes for evidence graph
            nodes = {node_id: GraphNode(
                id=node_id,
                type="claim",
                text=f"Claim {claim_id}",
                metadata={"claim_id": claim_id}
            ) for claim_id, node_id in graph.claim_nodes.items()}
            edges = graph.edges
        
        # Add nodes with styling based on type
        for node in nodes.values():
            node_id = self._escape_dot_id(node.id)
            label = self._escape_dot_string(node.text[:50] + "..." if len(node.text) > 50 else node.text)
            
            # Style based on node type
            if node.type == "source":
                color = "lightblue"
            elif node.type == "claim":
                color = "lightgreen"
            else:  # entity
                color = "lightyellow"
            
            dot_lines.append(f'  {node_id} [label="{label}", fillcolor={color}, style=filled];')
        
        dot_lines.append('')
        
        # Add edges with styling based on relationship type
        for edge in edges.values():
            source_id = self._escape_dot_id(edge.source_id)
            target_id = self._escape_dot_id(edge.target_id)
            
            # Style based on relationship type
            if edge.relationship_type == "SUPPORTS":
                color = "green"
                style = "solid"
            elif edge.relationship_type == "CONTRADICTS":
                color = "red"
                style = "solid"
            elif edge.relationship_type == "RELATES_TO":
                color = "blue"
                style = "dashed"
            else:
                color = "black"
                style = "solid"
            
            weight_label = f"{edge.weight:.2f}"
            dot_lines.append(
                f'  {source_id} -> {target_id} '
                f'[label="{edge.relationship_type}\\n{weight_label}", '
                f'color={color}, style={style}];'
            )
        
        dot_lines.append('}')
        
        return '\n'.join(dot_lines)
    
    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters."""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&apos;'))
    
    def _escape_dot_id(self, node_id: str) -> str:
        """Escape DOT node ID."""
        # Replace special characters with underscores
        import re
        return re.sub(r'[^a-zA-Z0-9_]', '_', node_id)
    
    def _escape_dot_string(self, text: str) -> str:
        """Escape DOT string literals."""
        return text.replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t')