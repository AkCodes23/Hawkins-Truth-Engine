"""
Unit tests for the GraphQueryInterface module.

This module tests the comprehensive query capabilities for both ClaimGraph and EvidenceGraph
structures, including node queries, relationship queries, graph metrics, and export functionality.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import Mock

from ..schemas import ClaimGraph, EvidenceGraph, GraphEdge, GraphNode
from .query_interface import GraphQueryInterface


class TestGraphQueryInterface:
    """Test cases for GraphQueryInterface functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.query_interface = GraphQueryInterface()
        
        # Create test nodes
        self.source_node = GraphNode(
            id="source:domain:example.com",
            type="source",
            text="Domain: example.com",
            metadata={"domain": "example.com", "trust_score": 0.8},
            confidence=0.8
        )
        
        self.claim_node1 = GraphNode(
            id="claim:C1",
            type="claim",
            text="Climate change is real",
            metadata={"claim_type": "factual", "support_status": "supported"},
            confidence=0.9
        )
        
        self.claim_node2 = GraphNode(
            id="claim:C2",
            type="claim",
            text="Climate change is not real",
            metadata={"claim_type": "factual", "support_status": "unsupported"},
            confidence=0.2
        )
        
        self.entity_node = GraphNode(
            id="entity:E1",
            type="entity",
            text="NASA",
            metadata={"entity_type": "ORG"},
            confidence=0.95
        )
        
        # Create test edges
        self.attribution_edge = GraphEdge(
            id="edge:1",
            source_id="claim:C1",
            target_id="source:domain:example.com",
            relationship_type="ATTRIBUTED_TO",
            weight=0.8,
            provenance={"method": "test"}
        )
        
        self.from_source_edge = GraphEdge(
            id="edge:2",
            source_id="source:domain:example.com",
            target_id="claim:C1",
            relationship_type="FROM_SOURCE",
            weight=0.8,
            provenance={"method": "test"}
        )
        
        self.mentions_edge = GraphEdge(
            id="edge:3",
            source_id="claim:C1",
            target_id="entity:E1",
            relationship_type="MENTIONS",
            weight=0.7,
            provenance={"method": "test"}
        )
        
        self.supports_edge = GraphEdge(
            id="evidence_edge:1",
            source_id="claim:C1",
            target_id="claim:C2",
            relationship_type="CONTRADICTS",
            weight=0.9,
            provenance={"similarity_score": 0.8, "evidence_strength": 0.9}
        )
        
        # Create test graphs
        self.claim_graph = ClaimGraph(
            nodes={
                "source:domain:example.com": self.source_node,
                "claim:C1": self.claim_node1,
                "claim:C2": self.claim_node2,
                "entity:E1": self.entity_node
            },
            edges={
                "edge:1": self.attribution_edge,
                "edge:2": self.from_source_edge,
                "edge:3": self.mentions_edge
            },
            metadata={"test": True}
        )
        
        self.evidence_graph = EvidenceGraph(
            claim_nodes={"C1": "claim:C1", "C2": "claim:C2"},
            edges={"evidence_edge:1": self.supports_edge},
            similarity_threshold=0.5,
            metadata={"test": True}
        )
    
    def test_find_claims_by_source_with_from_source_edges(self):
        """Test finding claims by source using FROM_SOURCE edges."""
        claims = self.query_interface.find_claims_by_source(
            self.claim_graph, "source:domain:example.com"
        )
        
        assert len(claims) == 1
        assert claims[0].id == "claim:C1"
        assert claims[0].type == "claim"
    
    def test_find_claims_by_source_with_attributed_to_edges(self):
        """Test finding claims by source using ATTRIBUTED_TO edges."""
        claims = self.query_interface.find_claims_by_source(
            self.claim_graph, "source:domain:example.com"
        )
        
        # Should find claim through both FROM_SOURCE and ATTRIBUTED_TO edges
        # but deduplicate
        assert len(claims) == 1
        assert claims[0].id == "claim:C1"
    
    def test_find_claims_by_source_nonexistent_source(self):
        """Test finding claims for a non-existent source."""
        claims = self.query_interface.find_claims_by_source(
            self.claim_graph, "source:nonexistent"
        )
        
        assert len(claims) == 0
    
    def test_find_claims_by_source_wrong_node_type(self):
        """Test finding claims when given node is not a source."""
        claims = self.query_interface.find_claims_by_source(
            self.claim_graph, "claim:C1"
        )
        
        assert len(claims) == 0
    
    def test_find_entities_in_claim(self):
        """Test finding entities mentioned in a claim."""
        entities = self.query_interface.find_entities_in_claim(
            self.claim_graph, "claim:C1"
        )
        
        assert len(entities) == 1
        assert entities[0].id == "entity:E1"
        assert entities[0].type == "entity"
        assert entities[0].text == "NASA"
    
    def test_find_entities_in_claim_nonexistent_claim(self):
        """Test finding entities for a non-existent claim."""
        entities = self.query_interface.find_entities_in_claim(
            self.claim_graph, "claim:nonexistent"
        )
        
        assert len(entities) == 0
    
    def test_find_entities_in_claim_wrong_node_type(self):
        """Test finding entities when given node is not a claim."""
        entities = self.query_interface.find_entities_in_claim(
            self.claim_graph, "source:domain:example.com"
        )
        
        assert len(entities) == 0
    
    def test_find_supporting_claims(self):
        """Test finding claims that support a given claim."""
        # Create a SUPPORTS edge for testing
        supports_edge = GraphEdge(
            id="evidence_edge:2",
            source_id="claim:C2",
            target_id="claim:C1",
            relationship_type="SUPPORTS",
            weight=0.8,
            provenance={"similarity_score": 0.7}
        )
        
        evidence_graph = EvidenceGraph(
            claim_nodes={"C1": "claim:C1", "C2": "claim:C2"},
            edges={"evidence_edge:2": supports_edge},
            similarity_threshold=0.5
        )
        
        supporting_claims = self.query_interface.find_supporting_claims(
            evidence_graph, "C1"
        )
        
        assert len(supporting_claims) == 1
        assert supporting_claims[0].id == "claim:C2"
        assert supporting_claims[0].confidence == 0.8
        assert "C2" in supporting_claims[0].metadata["claim_id"]
    
    def test_find_supporting_claims_bidirectional(self):
        """Test finding supporting claims works bidirectionally."""
        supports_edge = GraphEdge(
            id="evidence_edge:2",
            source_id="claim:C1",
            target_id="claim:C2",
            relationship_type="SUPPORTS",
            weight=0.8,
            provenance={"similarity_score": 0.7}
        )
        
        evidence_graph = EvidenceGraph(
            claim_nodes={"C1": "claim:C1", "C2": "claim:C2"},
            edges={"evidence_edge:2": supports_edge},
            similarity_threshold=0.5
        )
        
        supporting_claims = self.query_interface.find_supporting_claims(
            evidence_graph, "C2"
        )
        
        assert len(supporting_claims) == 1
        assert supporting_claims[0].id == "claim:C1"
    
    def test_find_contradicting_claims(self):
        """Test finding claims that contradict a given claim."""
        contradicting_claims = self.query_interface.find_contradicting_claims(
            self.evidence_graph, "C1"
        )
        
        assert len(contradicting_claims) == 1
        assert contradicting_claims[0].id == "claim:C2"
        assert contradicting_claims[0].confidence == 0.9
        assert "C2" in contradicting_claims[0].metadata["claim_id"]
    
    def test_find_contradicting_claims_bidirectional(self):
        """Test finding contradicting claims works bidirectionally."""
        contradicting_claims = self.query_interface.find_contradicting_claims(
            self.evidence_graph, "C2"
        )
        
        assert len(contradicting_claims) == 1
        assert contradicting_claims[0].id == "claim:C1"
    
    def test_find_related_claims(self):
        """Test finding claims that are topically related."""
        relates_edge = GraphEdge(
            id="evidence_edge:3",
            source_id="claim:C1",
            target_id="claim:C2",
            relationship_type="RELATES_TO",
            weight=0.6,
            provenance={"similarity_score": 0.5}
        )
        
        evidence_graph = EvidenceGraph(
            claim_nodes={"C1": "claim:C1", "C2": "claim:C2"},
            edges={"evidence_edge:3": relates_edge},
            similarity_threshold=0.5
        )
        
        related_claims = self.query_interface.find_related_claims(
            evidence_graph, "C1"
        )
        
        assert len(related_claims) == 1
        assert related_claims[0].id == "claim:C2"
        assert related_claims[0].confidence == 0.6
    
    def test_find_claims_nonexistent_claim_id(self):
        """Test finding relationships for non-existent claim ID."""
        supporting = self.query_interface.find_supporting_claims(
            self.evidence_graph, "nonexistent"
        )
        contradicting = self.query_interface.find_contradicting_claims(
            self.evidence_graph, "nonexistent"
        )
        related = self.query_interface.find_related_claims(
            self.evidence_graph, "nonexistent"
        )
        
        assert len(supporting) == 0
        assert len(contradicting) == 0
        assert len(related) == 0
    
    def test_calculate_node_centrality_claim_graph(self):
        """Test calculating node centrality for claim graph."""
        centrality = self.query_interface.calculate_node_centrality(self.claim_graph)
        
        # All nodes should have centrality scores
        assert len(centrality) == 4
        assert all(0.0 <= score <= 1.0 for score in centrality.values())
        
        # Source node should have high centrality (connected to claim)
        assert "source:domain:example.com" in centrality
        assert centrality["source:domain:example.com"] > 0.0
        
        # Claim node should have high centrality (connected to source and entity)
        assert "claim:C1" in centrality
        assert centrality["claim:C1"] > 0.0
    
    def test_calculate_node_centrality_evidence_graph(self):
        """Test calculating node centrality for evidence graph."""
        centrality = self.query_interface.calculate_node_centrality(self.evidence_graph)
        
        # Should have centrality for claim nodes
        assert len(centrality) == 2
        assert "claim:C1" in centrality
        assert "claim:C2" in centrality
        assert all(0.0 <= score <= 1.0 for score in centrality.values())
    
    def test_calculate_node_centrality_empty_graph(self):
        """Test calculating centrality for empty graph."""
        empty_graph = ClaimGraph()
        centrality = self.query_interface.calculate_node_centrality(empty_graph)
        
        assert len(centrality) == 0
    
    def test_calculate_graph_density_claim_graph(self):
        """Test calculating graph density for claim graph."""
        density = self.query_interface.calculate_graph_density(self.claim_graph)
        
        # 4 nodes, 3 edges, max possible = 4 * 3 = 12
        expected_density = 3 / 12
        assert density == expected_density
    
    def test_calculate_graph_density_evidence_graph(self):
        """Test calculating graph density for evidence graph."""
        density = self.query_interface.calculate_graph_density(self.evidence_graph)
        
        # 2 nodes, 1 edge, max possible = 2 * 1 = 2
        expected_density = 1 / 2
        assert density == expected_density
    
    def test_calculate_graph_density_single_node(self):
        """Test calculating density for graph with single node."""
        single_node_graph = ClaimGraph(
            nodes={"claim:C1": self.claim_node1},
            edges={}
        )
        
        density = self.query_interface.calculate_graph_density(single_node_graph)
        assert density == 0.0
    
    def test_get_subgraph_node_filter(self):
        """Test extracting subgraph with node filter."""
        # Filter to only include claim nodes
        node_filter = lambda node: node.type == "claim"
        
        subgraph = self.query_interface.get_subgraph(
            self.claim_graph, node_filter=node_filter
        )
        
        # Should only have claim nodes
        assert len(subgraph.nodes) == 2
        assert all(node.type == "claim" for node in subgraph.nodes.values())
        
        # Should have no edges (since source and entity nodes are filtered out)
        assert len(subgraph.edges) == 0
        
        # Should have subgraph metadata
        assert "subgraph_info" in subgraph.metadata
        assert subgraph.metadata["subgraph_info"]["original_nodes"] == 4
        assert subgraph.metadata["subgraph_info"]["filtered_nodes"] == 2
    
    def test_get_subgraph_edge_filter(self):
        """Test extracting subgraph with edge filter."""
        # Filter to only include high-weight edges
        edge_filter = lambda edge: edge.weight >= 0.8
        
        subgraph = self.query_interface.get_subgraph(
            self.claim_graph, edge_filter=edge_filter
        )
        
        # Should have all nodes
        assert len(subgraph.nodes) == 4
        
        # Should only have high-weight edges
        assert len(subgraph.edges) == 2  # attribution_edge and from_source_edge
        assert all(edge.weight >= 0.8 for edge in subgraph.edges.values())
    
    def test_get_subgraph_combined_filters(self):
        """Test extracting subgraph with both node and edge filters."""
        node_filter = lambda node: node.type in ["claim", "source"]
        edge_filter = lambda edge: edge.relationship_type in ["ATTRIBUTED_TO", "FROM_SOURCE"]
        
        subgraph = self.query_interface.get_subgraph(
            self.claim_graph, node_filter=node_filter, edge_filter=edge_filter
        )
        
        # Should have claim and source nodes only
        assert len(subgraph.nodes) == 3
        assert all(node.type in ["claim", "source"] for node in subgraph.nodes.values())
        
        # Should have attribution edges only
        assert len(subgraph.edges) == 2
        assert all(edge.relationship_type in ["ATTRIBUTED_TO", "FROM_SOURCE"] 
                  for edge in subgraph.edges.values())
    
    def test_traverse_graph_claim_graph(self):
        """Test graph traversal on claim graph."""
        result = self.query_interface.traverse_graph(
            self.claim_graph, "claim:C1", max_depth=2
        )
        
        assert "start_node" in result
        assert result["start_node"].id == "claim:C1"
        
        assert "paths" in result
        assert len(result["paths"]) > 0
        
        assert "discovered_nodes" in result
        assert len(result["discovered_nodes"]) > 1
        
        assert "traversal_stats" in result
        stats = result["traversal_stats"]
        assert "total_nodes_discovered" in stats
        assert "total_paths" in stats
        assert "relationship_distribution" in stats
    
    def test_traverse_graph_evidence_graph(self):
        """Test graph traversal on evidence graph."""
        result = self.query_interface.traverse_graph(
            self.evidence_graph, "claim:C1", max_depth=1
        )
        
        assert "start_node" in result
        assert result["start_node"].id == "claim:C1"
        
        assert "paths" in result
        assert "discovered_nodes" in result
        assert "traversal_stats" in result
    
    def test_traverse_graph_with_relationship_filter(self):
        """Test graph traversal with relationship type filter."""
        result = self.query_interface.traverse_graph(
            self.claim_graph, "claim:C1", max_depth=2, 
            relationship_types=["MENTIONS"]
        )
        
        # Should only follow MENTIONS edges
        for path in result["paths"]:
            assert path["relationship"] == "MENTIONS"
    
    def test_traverse_graph_nonexistent_start_node(self):
        """Test graph traversal with non-existent start node."""
        result = self.query_interface.traverse_graph(
            self.claim_graph, "nonexistent", max_depth=2
        )
        
        assert "error" in result
        assert "not found" in result["error"]
        assert len(result["paths"]) == 0
        assert len(result["discovered_nodes"]) == 0
    
    def test_export_json_claim_graph(self):
        """Test exporting claim graph as JSON."""
        json_output = self.query_interface.export_graph(self.claim_graph, "json")
        
        # Should be valid JSON
        data = json.loads(json_output)
        
        assert data["type"] == "ClaimGraph"
        assert "nodes" in data
        assert "edges" in data
        assert "metadata" in data
        assert "created_at" in data
        
        assert len(data["nodes"]) == 4
        assert len(data["edges"]) == 3
    
    def test_export_json_evidence_graph(self):
        """Test exporting evidence graph as JSON."""
        json_output = self.query_interface.export_graph(self.evidence_graph, "json")
        
        # Should be valid JSON
        data = json.loads(json_output)
        
        assert data["type"] == "EvidenceGraph"
        assert "claim_nodes" in data
        assert "edges" in data
        assert "similarity_threshold" in data
        assert "metadata" in data
        assert "created_at" in data
    
    def test_export_graphml(self):
        """Test exporting graph as GraphML."""
        graphml_output = self.query_interface.export_graph(self.claim_graph, "graphml")
        
        # Should contain GraphML structure
        assert '<?xml version="1.0"' in graphml_output
        assert '<graphml' in graphml_output
        assert '<graph id="G"' in graphml_output
        assert '</graphml>' in graphml_output
        
        # Should contain nodes and edges
        assert '<node id=' in graphml_output
        assert '<edge source=' in graphml_output
        
        # Should contain node and edge attributes
        assert 'node_type' in graphml_output
        assert 'edge_relationship' in graphml_output
    
    def test_export_dot(self):
        """Test exporting graph as DOT format."""
        dot_output = self.query_interface.export_graph(self.claim_graph, "dot")
        
        # Should contain DOT structure
        assert 'digraph G {' in dot_output
        assert 'rankdir=LR;' in dot_output
        assert '}' in dot_output
        
        # Should contain nodes and edges
        assert 'fillcolor=' in dot_output  # Node styling
        assert '->' in dot_output  # Edge syntax
        
        # Should contain relationship types
        assert 'ATTRIBUTED_TO' in dot_output
        assert 'FROM_SOURCE' in dot_output
        assert 'MENTIONS' in dot_output
    
    def test_export_unsupported_format(self):
        """Test exporting with unsupported format raises error."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            self.query_interface.export_graph(self.claim_graph, "unsupported")
    
    def test_xml_escaping(self):
        """Test XML character escaping."""
        test_text = 'Text with <tags> & "quotes" and \'apostrophes\''
        escaped = self.query_interface._escape_xml(test_text)
        
        assert '&lt;' in escaped
        assert '&gt;' in escaped
        assert '&amp;' in escaped
        assert '&quot;' in escaped
        assert '&apos;' in escaped
    
    def test_dot_id_escaping(self):
        """Test DOT ID escaping."""
        test_id = "claim:C1-test@domain.com"
        escaped = self.query_interface._escape_dot_id(test_id)
        
        # Should only contain valid DOT ID characters
        assert all(c.isalnum() or c == '_' for c in escaped)
    
    def test_dot_string_escaping(self):
        """Test DOT string escaping."""
        test_string = 'Text with "quotes" and\nnewlines\tand tabs'
        escaped = self.query_interface._escape_dot_string(test_string)
        
        assert '\\"' in escaped  # Escaped quotes
        assert '\\n' in escaped  # Escaped newlines
        assert '\\t' in escaped  # Escaped tabs


class TestGraphQueryInterfaceEdgeCases:
    """Test edge cases and error conditions for GraphQueryInterface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.query_interface = GraphQueryInterface()
    
    def test_empty_graphs(self):
        """Test operations on empty graphs."""
        empty_claim_graph = ClaimGraph()
        empty_evidence_graph = EvidenceGraph()
        
        # All query methods should return empty results
        assert len(self.query_interface.find_claims_by_source(empty_claim_graph, "any")) == 0
        assert len(self.query_interface.find_entities_in_claim(empty_claim_graph, "any")) == 0
        assert len(self.query_interface.find_supporting_claims(empty_evidence_graph, "any")) == 0
        assert len(self.query_interface.find_contradicting_claims(empty_evidence_graph, "any")) == 0
        assert len(self.query_interface.find_related_claims(empty_evidence_graph, "any")) == 0
        
        # Centrality should return empty dict
        assert len(self.query_interface.calculate_node_centrality(empty_claim_graph)) == 0
        assert len(self.query_interface.calculate_node_centrality(empty_evidence_graph)) == 0
        
        # Density should be 0
        assert self.query_interface.calculate_graph_density(empty_claim_graph) == 0.0
        assert self.query_interface.calculate_graph_density(empty_evidence_graph) == 0.0
    
    def test_malformed_node_ids(self):
        """Test handling of malformed node IDs."""
        # Create nodes with unusual ID formats
        node1 = GraphNode(id="malformed_id", type="claim", text="Test")
        node2 = GraphNode(id="another:malformed:id:with:colons", type="claim", text="Test")
        
        graph = ClaimGraph(nodes={"malformed_id": node1, "another:malformed:id:with:colons": node2})
        
        # Should handle gracefully
        centrality = self.query_interface.calculate_node_centrality(graph)
        assert len(centrality) == 2
        
        # Export should work
        json_output = self.query_interface.export_graph(graph, "json")
        assert json_output is not None
    
    def test_circular_relationships(self):
        """Test handling of circular relationships in graphs."""
        # Create nodes that reference each other in a circle
        node1 = GraphNode(id="claim:C1", type="claim", text="Claim 1")
        node2 = GraphNode(id="claim:C2", type="claim", text="Claim 2")
        node3 = GraphNode(id="claim:C3", type="claim", text="Claim 3")
        
        # Create circular edges: C1 -> C2 -> C3 -> C1
        edge1 = GraphEdge(id="e1", source_id="claim:C1", target_id="claim:C2", 
                         relationship_type="SUPPORTS", weight=0.5)
        edge2 = GraphEdge(id="e2", source_id="claim:C2", target_id="claim:C3", 
                         relationship_type="SUPPORTS", weight=0.5)
        edge3 = GraphEdge(id="e3", source_id="claim:C3", target_id="claim:C1", 
                         relationship_type="SUPPORTS", weight=0.5)
        
        evidence_graph = EvidenceGraph(
            claim_nodes={"C1": "claim:C1", "C2": "claim:C2", "C3": "claim:C3"},
            edges={"e1": edge1, "e2": edge2, "e3": edge3}
        )
        
        # Traversal should handle cycles gracefully
        result = self.query_interface.traverse_graph(evidence_graph, "claim:C1", max_depth=5)
        
        assert "error" not in result
        assert len(result["discovered_nodes"]) == 3  # Should discover all nodes
        assert result["traversal_stats"]["total_nodes_discovered"] == 3
    
    def test_high_degree_nodes(self):
        """Test performance with nodes that have many connections."""
        # Create a hub node connected to many other nodes
        hub_node = GraphNode(id="claim:hub", type="claim", text="Hub claim")
        nodes = {"claim:hub": hub_node}
        edges = {}
        
        # Create 50 connected nodes
        for i in range(50):
            node_id = f"claim:C{i}"
            edge_id = f"edge:{i}"
            
            nodes[node_id] = GraphNode(id=node_id, type="claim", text=f"Claim {i}")
            edges[edge_id] = GraphEdge(
                id=edge_id, source_id="claim:hub", target_id=node_id,
                relationship_type="SUPPORTS", weight=0.5
            )
        
        evidence_graph = EvidenceGraph(
            claim_nodes={f"C{i}": f"claim:C{i}" for i in range(50)} | {"hub": "claim:hub"},
            edges=edges
        )
        
        # Centrality calculation should complete
        centrality = self.query_interface.calculate_node_centrality(evidence_graph)
        
        # Hub node should have highest centrality
        assert centrality["claim:hub"] > 0.5
        assert centrality["claim:hub"] == max(centrality.values())
    
    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters in node text."""
        unicode_node = GraphNode(
            id="claim:unicode",
            type="claim",
            text="Claim with émojis 🌍 and spëcial çharacters"
        )
        
        special_node = GraphNode(
            id="claim:special",
            type="claim",
            text="Claim with <XML> & \"quotes\" and 'apostrophes'"
        )
        
        graph = ClaimGraph(
            nodes={"claim:unicode": unicode_node, "claim:special": special_node}
        )
        
        # Export should handle special characters
        json_output = self.query_interface.export_graph(graph, "json")
        assert json_output is not None
        
        graphml_output = self.query_interface.export_graph(graph, "graphml")
        assert graphml_output is not None
        assert "&lt;" in graphml_output  # XML should be escaped
        
        dot_output = self.query_interface.export_graph(graph, "dot")
        assert dot_output is not None
        assert '\\"' in dot_output  # Quotes should be escaped