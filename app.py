
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import time
import math
from algorithms import dfs, bfs, iterative_dfs, ucs, gbfs, astar, ida_star

# --- Pre-defined Hardcoded Graphs ---
GRAPHS = {
    "Simple Tree": {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    },
    "Complex Web": {
        '1': ['2', '3', '4'],
        '2': ['5', '6'],
        '3': ['6', '7'],
        '4': ['7', '8'],
        '5': ['9'],
        '6': ['9', '10'],
        '7': ['10'],
        '8': ['10'],
        '9': [],
        '10': []
    },
    "Cyclic Graph": {
        'A': ['B'],
        'B': ['C', 'E'],
        'C': ['D'],
        'D': ['B'],  # Cycle back to B
        'E': ['F'],
        'F': ['A']   # Cycle back to A
    },
    "Binary Search Tree": {
        '10': ['5', '15'],
        '5': ['2', '7'],
        '15': ['12', '20'],
        '2': [],
        '7': [],
        '12': [],
        '20': []
    }
}

# --- 1. Page Title & Settings ---
st.set_page_config(page_title="AI Algorithm Visualizer", layout="wide", initial_sidebar_state="collapsed")

# --- 2. Premium UI & Glassmorphism CSS ---
st.markdown("""
<style>
/* Premium animated gradient background */
.stApp {
    background: linear-gradient(-45deg, #050505, #101014, #1a1b26, #09090b);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glassmorphism specifically for the RIGHT column (Controls & Info) */
[data-testid="column"]:nth-of-type(2) {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

/* Input boxes styling */
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stTextArea textarea {
    background-color: rgba(0, 0, 0, 0.2) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
}

/* Select box styling */
[data-baseweb="select"] > div {
    background-color: rgba(0, 0, 0, 0.2) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: white !important;
}

/* Style buttons */
.stButton>button {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    backdrop-filter: blur(4px) !important;
    color: white !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    font-weight: 600;
}
.stButton>button:hover {
    background: rgba(255, 255, 255, 0.25) !important;
    box-shadow: 0 0 15px rgba(255, 255, 255, 0.2) !important;
    transform: translateY(-2px);
    border-color: rgba(255,255,255,0.5) !important;
}

/* Hide standard header, deploy button, and main menu completely */
header {visibility: hidden !important;}
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important;}
.stDeployButton {display: none !important;}

/* Remove default padding from main block so layout spreads nicely */
.block-container {
    padding-top: 3rem !important;
}

</style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if "steps" not in st.session_state:
    st.session_state.steps = []
if "step_index" not in st.session_state:
    st.session_state.step_index = 0
if "running" not in st.session_state:
    st.session_state.running = False
if "selected_graph_name" not in st.session_state:
    st.session_state.selected_graph_name = "Simple Tree"
    G = nx.DiGraph(GRAPHS["Simple Tree"])
    # Using Kamada Kawai for perfectly equal geometric distances
    st.session_state.graph_layout = nx.kamada_kawai_layout(G.to_undirected())

# --- UI Layout Split (70% Left, 30% Right) ---
col_left, col_right = st.columns([7, 3], gap="large")

with col_right:
    st.markdown("### Controls")
    
    import json
    
    # 1. Graph Selection
    graph_options = list(GRAPHS.keys()) + ["Custom Graph"]
    selected_graph_name = st.selectbox("Select Graph Map", graph_options)
    
    # Custom Graph Input Logic
    if selected_graph_name == "Custom Graph":
        with st.expander("📝 Custom Graph JSON Input", expanded=True):
            st.info("Format: Keys are node names, values are lists of neighbor nodes.")
            default_custom = '{\n  "A": ["B", "C"],\n  "B": ["D"],\n  "C": ["D"],\n  "D": []\n}'
            
            with st.form("custom_graph_form"):
                custom_json_str = st.text_area("JSON Graph Definition", default_custom, height=150)
                submit_custom = st.form_submit_button("✅ Load Custom Graph")
            
            try:
                parsed_custom = json.loads(custom_json_str)
                if not isinstance(parsed_custom, dict):
                    raise ValueError("Root must be a JSON object (dictionary).")
                for k, v in parsed_custom.items():
                    if not isinstance(v, list):
                        raise ValueError(f"Value for '{k}' must be a list of neighbors.")
                
                # Ensure all neighbors exist as keys (required for algorithms to avoid KeyError)
                for k, neighbors in parsed_custom.items():
                    for neighbor in neighbors:
                        if neighbor not in parsed_custom:
                            raise ValueError(f"Neighbor node '{neighbor}' is missing a root key definition. Please add `\"{neighbor}\": []`")
                            
                current_graph = parsed_custom
            except json.JSONDecodeError as e:
                st.error(f"Syntax Error: Invalid JSON Format - {str(e)}")
                st.stop()
            except Exception as e:
                st.error(f"Validation Error: {str(e)}")
                st.stop()
    else:
        current_graph = GRAPHS[selected_graph_name]
        
    current_graph_str = json.dumps(current_graph, sort_keys=True)
    
    if "current_graph_str" not in st.session_state:
        st.session_state.current_graph_str = current_graph_str
    
    # Process Graph Change (either selection changed, or custom JSON changed)
    if selected_graph_name != st.session_state.selected_graph_name or current_graph_str != st.session_state.current_graph_str:
        st.session_state.selected_graph_name = selected_graph_name
        st.session_state.current_graph_str = current_graph_str
        
        G = nx.DiGraph(current_graph)
        try:
            st.session_state.graph_layout = nx.kamada_kawai_layout(G.to_undirected())
        except Exception: # Fallback if graph is disconnected
            st.session_state.graph_layout = nx.spring_layout(G, seed=42)
            
        st.session_state.steps = []
        st.session_state.step_index = 0
        st.session_state.running = False
        st.rerun() # Refresh layout dynamically
        
    nodes_list = list(current_graph.keys())
    pos = st.session_state.graph_layout
    
    # Calculate geometric weights dynamically
    weights = {}
    for u, neighbors in current_graph.items():
        for v in neighbors:
            dist = math.hypot(pos[u][0] - pos[v][0], pos[u][1] - pos[v][1]) * 10
            weights[(u, v)] = max(1, int(round(dist)))
    
    # 2. Algorithm Selection
    algo = st.selectbox("Algorithm", ["BFS", "DFS", "IDS", "UCS", "GBFS", "A*", "IDA*"])
    
    # 3. Node Selection (Upgraded to Dropdowns for flawless UX)
    col_sn, col_gn = st.columns(2)
    with col_sn:
        start_node = st.selectbox("Start Node", nodes_list)
    with col_gn:
        # Default goal to the last node
        default_goal_index = len(nodes_list) - 1 if nodes_list else 0
        goal_node = st.selectbox("Goal Node", nodes_list, index=default_goal_index)
        
    # Calculate geometric heuristics dynamically towards goal
    heuristics = {}
    if goal_node in pos:
        for node in nodes_list:
            dist = math.hypot(pos[node][0] - pos[goal_node][0], pos[node][1] - pos[goal_node][1]) * 10
            heuristics[node] = max(0, int(round(dist)))
        
    if algo in ["IDS", "IDA*"]:
        max_depth = st.number_input("Max Depth", min_value=1, value=3, step=1)
    else:
        max_depth = None
        
    speed = st.slider("Playback Speed (s)", 0.1, 2.0, 0.5)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Run (Reset)", use_container_width=True):
            st.session_state.step_index = 0
            st.session_state.running = False
            if algo == "DFS":
                st.session_state.steps = dfs(current_graph, start_node, goal_node)
            elif algo == "BFS":
                st.session_state.steps = bfs(current_graph, start_node, goal_node)
            elif algo == "IDS":
                st.session_state.steps = iterative_dfs(current_graph, start_node, goal_node, max_depth)
            elif algo == "UCS":
                st.session_state.steps = ucs(current_graph, start_node, goal_node, weights)
            elif algo == "GBFS":
                st.session_state.steps = gbfs(current_graph, start_node, goal_node, heuristics)
            elif algo == "A*":
                st.session_state.steps = astar(current_graph, start_node, goal_node, weights, heuristics)
            elif algo == "IDA*":
                st.session_state.steps = ida_star(current_graph, start_node, goal_node, weights, heuristics)
                
    with col_btn2:
        if st.button("Auto Play", use_container_width=True):
            if not st.session_state.steps:
                # generate if empty
                if algo == "DFS":
                    st.session_state.steps = dfs(current_graph, start_node, goal_node)
                elif algo == "BFS":
                    st.session_state.steps = bfs(current_graph, start_node, goal_node)
                elif algo == "IDS":
                    st.session_state.steps = iterative_dfs(current_graph, start_node, goal_node, max_depth)
                elif algo == "UCS":
                    st.session_state.steps = ucs(current_graph, start_node, goal_node, weights)
                elif algo == "GBFS":
                    st.session_state.steps = gbfs(current_graph, start_node, goal_node, heuristics)
                elif algo == "A*":
                    st.session_state.steps = astar(current_graph, start_node, goal_node, weights, heuristics)
                elif algo == "IDA*":
                    st.session_state.steps = ida_star(current_graph, start_node, goal_node, weights, heuristics)
            st.session_state.running = True

    col_btn3, col_btn4 = st.columns(2)
    with col_btn3:
        if st.button("Next Step", use_container_width=True):
            st.session_state.running = False
            if st.session_state.steps and st.session_state.step_index < len(st.session_state.steps) - 1:
                st.session_state.step_index += 1
                
    with col_btn4:
        if st.button("Clear", use_container_width=True):
            st.session_state.steps = []
            st.session_state.step_index = 0
            st.session_state.running = False

    st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.1); margin: 15px 0;'>", unsafe_allow_html=True)

    with st.expander("View Graph Data Structure"):
        st.json(current_graph)
        
    if st.session_state.steps and st.session_state.step_index < len(st.session_state.steps):
        st.markdown("### Step Info")
        current_step_data = st.session_state.steps[st.session_state.step_index]
        
        st.write(f"**Step:** `{st.session_state.step_index + 1} / {len(st.session_state.steps)}`")
        st.write(f"**Current Node:** `{current_step_data.get('current')}`")
        
        path_str = " → ".join(current_step_data.get("path", []))
        st.write(f"**Path:** `{path_str}`")
        
        if "depth" in current_step_data:
            st.write(f"**Depth Limit:** `{current_step_data['depth']}`")
        if "cost" in current_step_data:
            st.write(f"**g(n) Cost:** `{current_step_data['cost']}`")
        if "heuristic" in current_step_data:
            st.write(f"**h(n) Heuristic:** `{current_step_data['heuristic']}`")
        if "f_cost" in current_step_data:
            st.write(f"**f(n) Cost:** `{current_step_data['f_cost']}`")
        if "f_limit" in current_step_data:
            st.write(f"**f(n) Limit:** `{current_step_data['f_limit']}`")
            
        st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.1); margin: 15px 0;'>", unsafe_allow_html=True)
        st.markdown("### Debug Console")
        st.info(f"Frontier: {current_step_data.get('frontier', [])}")
        st.success(f"Visited: {current_step_data.get('visited', [])}")
    else:
        st.info("Configure parameters and click Run to begin visualization.")


with col_left:
    st.title("AI Algorithm Visualizer")
    
    # Establish graph and position
    G = nx.DiGraph(current_graph)
    pos = st.session_state.graph_layout
    
    current_node = None
    frontier = []
    visited = []
    path = []
    
    # Get current state
    if st.session_state.steps and st.session_state.step_index < len(st.session_state.steps):
        current_step_data = st.session_state.steps[st.session_state.step_index]
        current_node = current_step_data.get("current")
        frontier = current_step_data.get("frontier", [])
        visited = current_step_data.get("visited", [])
        path = current_step_data.get("path", [])
        
    # --- 5. Graph Visualization ---
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_alpha(0.0) # Transparent figure background
    ax.patch.set_alpha(0.0)  # Transparent axes background
    ax.axis("off")
    
    # Identify edges that are in the active path for glowing white effect
    edge_colors = []
    edge_widths = []
    path_edges = list(zip(path[:-1], path[1:]))
    
    for u, v in G.edges():
        if (u, v) in path_edges:
            edge_colors.append((1.0, 1.0, 1.0, 1.0)) # Solid White
            edge_widths.append(3.0)
        else:
            edge_colors.append((1.0, 1.0, 1.0, 0.2)) # Translucent White
            edge_widths.append(1.5)
            
    # Professional glassmorphic colors (RGBA tuples for transparency)
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == current_node:
            node_colors.append((0.91, 0.30, 0.24, 0.7)) # Soft Red (Current)
            node_sizes.append(2500)
        elif node in visited:
            node_colors.append((0.18, 0.80, 0.44, 0.5)) # Sage Green (Visited)
            node_sizes.append(1800)
        elif node in frontier:
            node_colors.append((0.95, 0.77, 0.06, 0.5)) # Warm Gold (Frontier)
            node_sizes.append(1800)
        else:
            node_colors.append((0.29, 0.56, 0.89, 0.4)) # Soft Slate Blue (Unvisited)
            node_sizes.append(1500)
            
    # Draw Nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax, 
        node_color=node_colors, 
        node_size=node_sizes, 
        edgecolors=(1.0, 1.0, 1.0, 0.7), 
        linewidths=2.5
    )
                           
    # Draw Edges
    nx.draw_networkx_edges(
        G, pos, ax=ax, 
        edge_color=edge_colors, 
        width=edge_widths,
        node_size=node_sizes,
        arrowsize=25, 
        arrowstyle="-|>", 
        connectionstyle="arc3,rad=0.15"
    )
    
    # Draw Edge Labels (Weights)
    edge_labels = {edge: str(weight) for edge, weight in weights.items()}
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax,
        edge_labels=edge_labels,
        font_color="white",
        font_size=10,
        label_pos=0.3,
        bbox=dict(facecolor="#101014", edgecolor="none", alpha=0.7, pad=2)
    )
                           
    # Draw Labels
    nx.draw_networkx_labels(
        G, pos, ax=ax, 
        font_color="white", 
        font_size=15, 
        font_weight="bold"
    )
    
    # Render Plot
    st.pyplot(fig)

# --- Animation Logic ---
if st.session_state.running:
    if st.session_state.step_index < len(st.session_state.steps) - 1:
        time.sleep(speed)
        st.session_state.step_index += 1
        st.rerun()
    else:
        st.session_state.running = False
