import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
import sys
from graph_tool.all import Graph, graph_draw

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag')))
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../rag')))
import ragSqlUtils as sq

def plot_3d(nodes, edges, title):
    # Generate unique IDs for positions by prefixing "entity_" and "tag_"
    positions = {
        f"entity_{node['name']}": np.random.rand(3) * 10 for node in nodes if node['type'] == 'entity'
    }
    positions.update({
        f"tag_{node['name']}": np.random.rand(3) * 10 for node in nodes if node['type'] == 'tag'
    })

    # Extract edges with unique identifiers
    edge_x, edge_y, edge_z = [], [], []
    for edge in edges:
        source_key = f"entity_{edge['source']}"  # Entity IDs prefixed
        target_key = f"tag_{edge['target']}"     # Tag IDs prefixed
        x0, y0, z0 = positions[source_key]
        x1, y1, z1 = positions[target_key]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    # Split nodes into entities and tags using unique identifiers
    entity_x, entity_y, entity_z = [], [], []
    tag_x, tag_y, tag_z = [], [], []

    for node in entity_nodes:
        x, y, z = positions[f"entity_{node['name']}"]
        entity_x.append(x)
        entity_y.append(y)
        entity_z.append(z)

    for node in tag_nodes:
        x, y, z = positions[f"tag_{node['name']}"]
        tag_x.append(x)
        tag_y.append(y)
        tag_z.append(z)

    # Plot edges
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(width=2, color='gray'),
        hoverinfo='none'
    )

    # Plot entities (red)
    entity_trace = go.Scatter3d(
        x=entity_x, y=entity_y, z=entity_z,
        mode='markers+text',
        marker=dict(size=10, color='red'),
        text=[node["name"] for node in entity_nodes],
        textposition="top center",
        name="Entities"
    )

    # Plot tags (blue)
    tag_trace = go.Scatter3d(
        x=tag_x, y=tag_y, z=tag_z,
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=[node["name"] for node in tag_nodes],
        textposition="top center",
        name="Tags"
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, entity_trace, tag_trace])
    fig.update_layout(
        title=title,
        showlegend=True,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        )
    )

    # Show the plot
    fig.show()


def plot_2d(nodes, edges,title):

    # Create the graph
    G = nx.Graph()

    # Add nodes with attributes
    for node in nodes:
        if node["type"] == "entity":
            G.add_node(node["name"], type="entity", color="red")
        else:
            G.add_node(node["name"], type="tag", color="blue")

    # Add edges
    for edge in edges:
        G.add_edge(edge["source"], edge["target"])

    # Extract node colors
    node_colors = [G.nodes[node]["color"] for node in G.nodes]

    # Draw the graph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.1)  # Spring layout for positioning
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=500,
        font_size=10,
        edge_color="gray",
    )
    plt.title(title)
    plt.show()

def plot_2dgt(nodes, edges, titles):
    # Create graph
    g = Graph(directed=False)

    # Map names to vertices
    vertex_map = {}

    # Add vertices
    for node in nodes:
        v = g.add_vertex()
        vertex_map[node["name"]] = v

    # Add edges
    for edge in edges:
        g.add_edge(vertex_map[edge["source"]], vertex_map[edge["target"]])

    # Draw graph
    graph_draw(g, output_size=(1000, 1000), vertex_text=g.vertex_index, output="graph.pdf")

def createGraph(nodes,edges,title):
    # Create the graph
    G = nx.Graph()

    # Add nodes
    for node in nodes:
        G.add_node(node["name"], type=node["type"])

    # Add edges
    for edge in edges:
        G.add_edge(edge["source"], edge["target"])

    # Export to GraphML
    file = f"{title}.graphml"
    nx.write_graphml(G, file)
    print(f"Graph exported to {file}. Open this file in Gephi.")
    

cs = "sqlite:///../projects/ksk.db"
db = sq.DatabaseUtility(cs)

entity_data = db.search(sq.Item)
tag_data = db.search(sq.Tag)

# JSON for nodes as entities
entity_nodes = [{"id": e.id, "name": e.name} for e in entity_data]
entity_edges = []
for e in entity_nodes:
    tags = db.get_item_tags(e["id"])
    entity_edges.extend(
    {"source": e["name"], "target": t}
        for t in tags
    )

# JSON for nodes as tags
tag_nodes = [{"id": t.id, "name": t.name} for t in tag_data]

# Combine nodes
allNodes = [{"id": e["id"], "name": e["name"], "type": "entity"} for e in entity_nodes] + \
        [{"id": t["id"], "name": t["name"], "type": "tag"} for t in tag_nodes]



#plot_3d(allNodes, entity_edges, "Entities and Tags")
plot_2d(allNodes, entity_edges, "Entities and Tags")
#plot_2dgt(allNodes, entity_edges, "Entities and Tags")
#createGraph(allNodes, entity_edges, "EntitiesAndTags")
