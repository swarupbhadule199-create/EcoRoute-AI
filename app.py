
import heapq
import time

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EcoRoute-AI",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 EcoRoute-AI")
st.subheader("AI-Based Fuel-Efficient Route Planning")
st.write(
    "Select your source and destination, then compare all three "
    "search algorithms under the selected simulated traffic conditions."
)


# =========================================================
# 2. PROJECT CONFIGURATION
# =========================================================

TIME_WEIGHT = 0.40
FUEL_WEIGHT = 0.60
TIME_REFERENCE = 60.0
FUEL_REFERENCE = 5.0

LOCATIONS = list("ABCDEFGHIJ")

ROADS = [
    ("A", "B", 10, 60, 6.0),
    ("A", "C", 15, 75, 6.5),
    ("A", "D", 20, 80, 7.0),
    ("B", "E", 12, 55, 6.0),
    ("B", "F", 18, 70, 6.5),
    ("C", "E", 8, 50, 5.5),
    ("C", "F", 10, 65, 6.0),
    ("C", "G", 20, 80, 7.0),
    ("D", "G", 12, 60, 6.0),
    ("E", "H", 15, 70, 6.0),
    ("F", "H", 10, 60, 5.5),
    ("F", "I", 15, 75, 6.0),
    ("G", "I", 10, 65, 6.0),
    ("H", "J", 12, 70, 5.5),
    ("I", "J", 8, 50, 5.0),
]

POSITIONS = {
    "A": (0, 2),
    "B": (1, 4),
    "C": (1, 1),
    "D": (1, -1),
    "E": (3, 4),
    "F": (3, 2),
    "G": (3, -1),
    "H": (5, 4),
    "I": (5, 1),
    "J": (7, 2),
}


# =========================================================
# 3. GRAPH AND TRAFFIC MODEL
# =========================================================

def build_graph(traffic, congest_fh=False):
    graph = nx.Graph()
    graph.add_nodes_from(LOCATIONS)

    traffic_factors = {
        "Low": (1.00, 1.00),
        "Medium": (0.80, 1.05),
        "High": (0.55, 1.15),
    }

    speed_factor, fuel_factor = traffic_factors[traffic]

    for u, v, distance, speed, fuel_rate in ROADS:
        effective_speed = speed * speed_factor
        effective_fuel_rate = fuel_rate * fuel_factor

        if congest_fh and {u, v} == {"F", "H"}:
            effective_speed *= 0.55
            effective_fuel_rate *= 1.15

        travel_time = 60 * distance / effective_speed
        fuel = distance * effective_fuel_rate / 100

        cost = (
            TIME_WEIGHT * travel_time / TIME_REFERENCE
            + FUEL_WEIGHT * fuel / FUEL_REFERENCE
        )

        graph.add_edge(
            u, v,
            distance=distance,
            speed=effective_speed,
            fuel_rate=effective_fuel_rate,
            time=travel_time,
            fuel=fuel,
            cost=cost
        )

    return graph


# =========================================================
# 4. SEARCH ALGORITHMS
# =========================================================

def reconstruct_path(parent, destination):
    path = [destination]

    while path[-1] in parent:
        path.append(parent[path[-1]])

    return path[::-1]


def bfs(graph, source, destination):
    queue = [source]
    visited = {source}
    parent = {}
    expanded = 0

    while queue:
        node = queue.pop(0)
        expanded += 1

        if node == destination:
            return reconstruct_path(parent, destination), expanded

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append(neighbor)

    return None, expanded


def greedy_search(graph, source, destination):
    remaining_distance = nx.single_source_dijkstra_path_length(
        graph, destination, weight="distance"
    )

    frontier = [(remaining_distance[source], source)]
    visited = set()
    parent = {}
    expanded = 0

    while frontier:
        _, node = heapq.heappop(frontier)

        if node in visited:
            continue

        visited.add(node)
        expanded += 1

        if node == destination:
            return reconstruct_path(parent, destination), expanded

        for neighbor in graph.neighbors(node):
            if neighbor not in visited and neighbor not in parent:
                parent[neighbor] = node
                heapq.heappush(
                    frontier, (remaining_distance[neighbor], neighbor)
                )

    return None, expanded


def astar(graph, source, destination):
    min_cost_per_km = min(
        data["cost"] / data["distance"]
        for _, _, data in graph.edges(data=True)
    )

    remaining_distance = nx.single_source_dijkstra_path_length(
        graph, destination, weight="distance"
    )

    def heuristic(node):
        return min_cost_per_km * remaining_distance[node]

    frontier = [(heuristic(source), 0.0, source)]
    best_g = {source: 0.0}
    parent = {}
    expanded = 0

    while frontier:
        _, current_g, node = heapq.heappop(frontier)

        if current_g > best_g.get(node, float("inf")):
            continue

        expanded += 1

        if node == destination:
            return reconstruct_path(parent, destination), expanded

        for neighbor in graph.neighbors(node):
            new_g = current_g + graph[node][neighbor]["cost"]

            if new_g < best_g.get(neighbor, float("inf")):
                best_g[neighbor] = new_g
                parent[neighbor] = node

                heapq.heappush(
                    frontier,
                    (new_g + heuristic(neighbor), new_g, neighbor)
                )

    return None, expanded


ALGORITHMS = {
    "BFS": bfs,
    "Greedy Best-First": greedy_search,
    "A*": astar,
}


# =========================================================
# 5. ROUTE METRICS
# =========================================================

def calculate_metrics(graph, path):
    distance = 0.0
    fuel = 0.0
    travel_time = 0.0

    for u, v in zip(path[:-1], path[1:]):
        edge = graph[u][v]
        distance += edge["distance"]
        fuel += edge["fuel"]
        travel_time += edge["time"]

    combined_cost = (
        TIME_WEIGHT * travel_time / TIME_REFERENCE
        + FUEL_WEIGHT * fuel / FUEL_REFERENCE
    )

    return {
        "Route": " → ".join(path),
        "Roads": len(path) - 1,
        "Distance (km)": round(distance, 2),
        "Fuel (L)": round(fuel, 3),
        "Time (min)": round(travel_time, 2),
        "Combined Cost": round(combined_cost, 4),
    }


# =========================================================
# 6. ROAD-NETWORK VISUALIZATION
# =========================================================

def draw_network(graph, source=None, destination=None, route=None):
    fig = go.Figure()

    # Draw all road connections.
    for u, v, data in graph.edges(data=True):
        x1, y1 = POSITIONS[u]
        x2, y2 = POSITIONS[v]

        fig.add_trace(go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode="lines",
            line=dict(color="#cbd5e1", width=2),
            hoverinfo="text",
            text=[f"{u}–{v}: {data['distance']} km"] * 2,
            showlegend=False
        ))

        fig.add_annotation(
            x=(x1 + x2) / 2,
            y=(y1 + y2) / 2,
            text=f"{data['distance']} km",
            showarrow=False,
            font=dict(size=10, color="#000000"),
            bgcolor="rgba(255,255,255,0.8)"
        )

    # Highlight the selected route.
    if route:
        for u, v in zip(route[:-1], route[1:]):
            x1, y1 = POSITIONS[u]
            x2, y2 = POSITIONS[v]

            fig.add_trace(go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                line=dict(color="#ef4444", width=5),
                hoverinfo="skip",
                showlegend=False
            ))

    node_colors = []

    for node in LOCATIONS:
        if node == source:
            node_colors.append("#22c55e")
        elif node == destination:
            node_colors.append("#ef4444")
        else:
            node_colors.append("#010100")

    fig.add_trace(go.Scatter(
        x=[POSITIONS[n][0] for n in LOCATIONS],
        y=[POSITIONS[n][1] for n in LOCATIONS],
        mode="markers+text",
        text=LOCATIONS,
        textposition="middle center",
        marker=dict(
            size=35,
            color=node_colors,
            line=dict(color="#334155", width=1.5)
        ),
        hovertemplate="Location %{text}<extra></extra>",
        showlegend=False
    ))

    fig.update_layout(
        height=470,
        margin=dict(l=10, r=10, t=35, b=10),
        title="Simulated Road Network",
        xaxis=dict(visible=False, range=[-0.5, 7.5]),
        yaxis=dict(visible=False, range=[-1.5, 4.7]),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    return fig


# =========================================================
# 7. USER INPUTS — DROPDOWNS
# =========================================================

st.sidebar.header("Journey Planner")

source = st.sidebar.selectbox(
    "Choose Source",
    LOCATIONS,
    index=0
)

destination = st.sidebar.selectbox(
    "Choose Destination",
    LOCATIONS,
    index=9
)

traffic = st.sidebar.selectbox(
    "Traffic Intensity",
    ["Low", "Medium", "High"]
)

congest_fh = st.sidebar.checkbox(
    "Apply extra congestion to road F–H"
)

st.sidebar.caption(
    "All three algorithms run automatically for the selected journey."
)

# Build graph using the selected traffic settings.
graph = build_graph(traffic, congest_fh)


# =========================================================
# 8. DISPLAY THE FULL ROAD NETWORK
# =========================================================

st.header("Road Network")

st.plotly_chart(
    draw_network(graph, source, destination),
    use_container_width=True,
    key="full_road_network"
)

st.caption(
    "Green = source. Red = destination. "
    "The red route lines appear in the individual algorithm maps below."
)


# =========================================================
# 9. RUN ALL ALGORITHMS AUTOMATICALLY
# =========================================================

if source == destination:
    st.warning("Please select different source and destination locations.")

else:
    results = []
    routes = {}

    for name, search_function in ALGORITHMS.items():
        start_time = time.perf_counter()

        route, expanded_nodes = search_function(
            graph, source, destination
        )

        execution_ms = (time.perf_counter() - start_time) * 1000

        if route is None:
            continue

        routes[name] = route

        results.append({
            "Algorithm": name,
            **calculate_metrics(graph, route),
            "Expanded Nodes": expanded_nodes,
            "Execution Time (ms)": round(execution_ms, 4)
        })

    if results:
        results_df = pd.DataFrame(results)

        best = results_df.loc[
            results_df["Combined Cost"].idxmin()
        ]

        st.divider()
        st.header(f"Route Results: {source} → {destination}")

        st.success(
            f"Lowest combined cost: {best['Algorithm']} "
            f"with route {best['Route']}"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Distance", f"{best['Distance (km)']} km")
        c2.metric("Estimated Fuel", f"{best['Fuel (L)']} L")
        c3.metric("Estimated Time", f"{best['Time (min)']} min")
        c4.metric("Combined Cost", str(best["Combined Cost"]))

        st.subheader("Comparison of All Three Algorithms")

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        # Each algorithm gets its own route map.
        st.header("Individual Route Maps")

        for name, route in routes.items():
            with st.expander(
                f"{name}: {' → '.join(route)}",
                expanded=True
            ):
                st.plotly_chart(
                    draw_network(
                        graph, source, destination, route
                    ),
                    use_container_width=True,
                    key=f"route_map_{name}"
                )

        # Comparison charts.
        st.header("Performance Comparisons")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Fuel Consumption")
            st.bar_chart(
                results_df.set_index("Algorithm")[["Fuel (L)"]]
            )

        with col2:
            st.subheader("Estimated Travel Time")
            st.bar_chart(
                results_df.set_index("Algorithm")[["Time (min)"]]
            )

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Search Efficiency — Expanded Nodes")
            st.bar_chart(
                results_df.set_index("Algorithm")[["Expanded Nodes"]]
            )

        with col4:
            st.subheader("Combined Route Cost")
            st.bar_chart(
                results_df.set_index("Algorithm")[["Combined Cost"]]
            )

        st.download_button(
            "Download Results as CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name=f"EcoRoute_{source}_to_{destination}.csv",
            mime="text/csv"
        )

st.caption(
    "This application uses a fictional road network and simulated "
    "traffic conditions. It does not provide live GPS or traffic data."
)
