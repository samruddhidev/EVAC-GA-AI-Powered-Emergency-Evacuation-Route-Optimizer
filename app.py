import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="EVAC-GA | Emergency Evacuation Optimizer",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f7fbff 0%, #eef6ff 48%, #fff7f7 100%);
    color: #17324d;
}
.block-container { max-width: 1500px; padding-top: 1.1rem; padding-bottom: 3rem; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f2f8ff 100%);
    border-right: 1px solid #dbeafe;
}
.hero {
    padding: 28px 32px;
    border-radius: 22px;
    background: linear-gradient(110deg, #ffffff 0%, #eef7ff 68%, #fff2f2 100%);
    border: 1px solid #cfe3f7;
    box-shadow: 0 14px 42px rgba(30, 64, 175, .10);
    margin-bottom: 18px;
}
.hero h1 { font-size: 40px; margin: 8px 0 2px; color: #12395d; letter-spacing: -1px; }
.hero p { color: #55718d; font-size: 16px; margin: 5px 0; }
.status {
    display: inline-block; padding: 6px 12px; border-radius: 999px;
    color: #b91c1c; background: #fff1f2; border: 1px solid #fecdd3;
    font-size: 12px; font-weight: 750; letter-spacing: .35px;
}
.card {
    padding: 19px 21px; border-radius: 17px; background: rgba(255,255,255,.88);
    border: 1px solid #dbeafe; margin: 13px 0; box-shadow: 0 8px 25px rgba(30,64,175,.06);
}
.kpi {
    padding: 16px 17px; border-radius: 15px; background: #ffffff;
    border: 1px solid #dbeafe; min-height: 105px;
    box-shadow: 0 8px 22px rgba(30,64,175,.07);
}
.kpi-label { color: #66809a; font-size: 12px; }
.kpi-value { font-size: 26px; font-weight: 780; margin-top: 5px; color: #153e63; }
.kpi-note { color: #2563eb; font-size: 12px; margin-top: 3px; }
.muted { color: #617b94; font-size: 13px; }
.legend { display:flex; gap:15px; flex-wrap:wrap; color:#55718d; font-size:12px; padding:8px 0; }
.legend b { font-weight:650; }
.stButton>button {
    border-radius: 10px; border: 1px solid #dc2626;
    background: linear-gradient(90deg,#dc2626,#ef4444); color:white;
    font-weight:750; box-shadow: 0 7px 20px rgba(220,38,38,.16);
}
.stButton>button:hover { border-color:#2563eb; background:linear-gradient(90deg,#1d4ed8,#2563eb); }
div[data-testid="stMetric"] { background:#ffffff; border:1px solid #dbeafe; border-radius:14px; padding:10px; }
[data-testid="stTabs"] button { color:#34546f; }
</style>
""",unsafe_allow_html=True)


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class RouteMetrics:
    distance: float
    time: float
    congestion: float
    risk: float
    blocked_edges: int
    crowd_exposure: float
    fitness: float

# -----------------------------
# Network generation
# -----------------------------
def build_city(seed=42, rows=9, cols=11):
    random.seed(seed)
    np.random.seed(seed)

    G = nx.grid_2d_graph(rows, cols)

    for node in G.nodes:
        r, c = node
        G.nodes[node]["x"] = c
        G.nodes[node]["y"] = r

    for u, v in G.edges:
        base_distance = random.uniform(0.7, 1.5)
        speed = random.uniform(25, 45)
        congestion = random.uniform(0.05, 0.85)
        safety = random.uniform(0.05, 0.55)
        capacity = random.randint(40, 160)

        G.edges[u, v]["distance"] = base_distance
        G.edges[u, v]["speed"] = speed
        G.edges[u, v]["congestion"] = congestion
        G.edges[u, v]["safety"] = safety
        G.edges[u, v]["capacity"] = capacity
        G.edges[u, v]["blocked"] = False

    return G

def apply_scenario(G, blockage_rate, congestion_level, safety_level, seed):
    random.seed(seed + 101)
    np.random.seed(seed + 101)

    for u, v in G.edges:
        G.edges[u, v]["blocked"] = random.random() < blockage_rate

        base_cong = G.edges[u, v]["congestion"]
        G.edges[u, v]["congestion"] = min(
            1.0, 0.35 * base_cong + 0.65 * congestion_level * random.uniform(0.75, 1.15)
        )

        base_safety = G.edges[u, v]["safety"]
        G.edges[u, v]["safety"] = min(
            1.0, 0.35 * base_safety + 0.65 * safety_level * random.uniform(0.75, 1.20)
        )

def safe_edge(G, u, v):
    return not G.edges[u, v]["blocked"]

def available_neighbors(G, node):
    return [n for n in G.neighbors(node) if safe_edge(G, node, n)]

def nearest_available_path(G, start, goal):
    H = G.copy()
    remove = [(u, v) for u, v, d in H.edges(data=True) if d["blocked"]]
    H.remove_edges_from(remove)
    try:
        return nx.shortest_path(H, start, goal, weight="distance")
    except nx.NetworkXNoPath:
        return None

# -----------------------------
# Route scoring
# -----------------------------
def edge_cost(G, u, v, weights):
    e = G.edges[u, v]

    if e["blocked"]:
        return 1e6

    distance = e["distance"]
    time = distance / max(e["speed"], 1) * (1 + 2.2 * e["congestion"])
    risk = e["safety"] * (1 + 1.7 * e["congestion"])
    crowd = e["congestion"] * (1.0 + distance)

    return (
        weights["distance"] * distance
        + weights["time"] * time
        + weights["congestion"] * crowd
        + weights["risk"] * risk
    )

def route_metrics(G, route, weights):
    if not route or len(route) < 2:
        return RouteMetrics(9999, 9999, 9999, 9999, 99, 9999, 1e9)

    distance = 0
    time = 0
    congestion = 0
    risk = 0
    blocked = 0
    crowd = 0

    for u, v in zip(route[:-1], route[1:]):
        if not G.has_edge(u, v):
            return RouteMetrics(9999, 9999, 9999, 9999, 99, 9999, 1e9)
        e = G.edges[u, v]
        distance += e["distance"]
        time += e["distance"] / max(e["speed"], 1) * (1 + 2.2 * e["congestion"])
        congestion += e["congestion"]
        risk += e["safety"] * (1 + 1.7 * e["congestion"])
        crowd += e["congestion"] * (1 + e["distance"])
        blocked += int(e["blocked"])

    n = max(1, len(route) - 1)
    congestion /= n
    risk /= n

    fitness = (
        weights["distance"] * distance
        + weights["time"] * time
        + weights["congestion"] * crowd
        + weights["risk"] * risk
        + weights["blocked"] * blocked
    )

    return RouteMetrics(distance, time, congestion, risk, blocked, crowd, fitness)

# -----------------------------
# Genetic Algorithm
# -----------------------------
class EvacuationGA:
    def __init__(
        self,
        G,
        start,
        goal,
        weights,
        population_size=80,
        generations=100,
        mutation_rate=0.15,
        crossover_rate=0.85,
        elite_size=6,
        seed=42
    ):
        self.G = G
        self.start = start
        self.goal = goal
        self.weights = weights
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.rng = random.Random(seed)
        self.max_nodes = len(G.nodes)

    def random_path(self):
        current = self.start
        path = [current]
        visited = {current}

        for _ in range(self.max_nodes):
            if current == self.goal:
                return path

            neighbors = [n for n in available_neighbors(self.G, current) if n not in visited]

            if not neighbors:
                return None

            # Bias movement toward goal while retaining exploration.
            scored = []
            for n in neighbors:
                dist_to_goal = abs(n[0] - self.goal[0]) + abs(n[1] - self.goal[1])
                edge = self.G.edges[current, n]
                score = dist_to_goal + 3 * edge["congestion"] + 2 * edge["safety"]
                scored.append((score, n))

            scored.sort(key=lambda x: x[0])
            top = scored[:max(2, min(5, len(scored)))]
            current = self.rng.choice(top)[1]
            path.append(current)
            visited.add(current)

        return None

    def initial_population(self):
        pop = []
        attempts = 0

        while len(pop) < self.population_size and attempts < self.population_size * 50:
            p = self.random_path()
            if p and p[-1] == self.goal:
                pop.append(p)
            attempts += 1

        # Guarantee at least one feasible path if available.
        fallback = nearest_available_path(self.G, self.start, self.goal)
        if fallback and fallback not in pop:
            pop.append(fallback)

        return pop

    def repair(self, path):
        if not path:
            return None

        # Remove loops while preserving order.
        result = []
        seen = {}
        for node in path:
            if node in seen:
                result = result[:seen[node] + 1]
                seen = {n: i for i, n in enumerate(result)}
            else:
                seen[node] = len(result)
                result.append(node)

        if result[0] != self.start:
            result.insert(0, self.start)

        # If a broken segment exists, connect it with shortest feasible subpath.
        repaired = [result[0]]
        for a, b in zip(result[:-1], result[1:]):
            if self.G.has_edge(a, b) and safe_edge(self.G, a, b):
                repaired.append(b)
            else:
                H = self.G.copy()
                H.remove_edges_from([(u, v) for u, v, d in H.edges(data=True) if d["blocked"]])
                try:
                    sub = nx.shortest_path(H, a, b, weight="distance")
                    repaired.extend(sub[1:])
                except nx.NetworkXNoPath:
                    return None

        if repaired[-1] != self.goal:
            H = self.G.copy()
            H.remove_edges_from([(u, v) for u, v, d in H.edges(data=True) if d["blocked"]])
            try:
                sub = nx.shortest_path(H, repaired[-1], self.goal, weight="distance")
                repaired.extend(sub[1:])
            except nx.NetworkXNoPath:
                return None

        return repaired

    def fitness(self, path):
        return route_metrics(self.G, path, self.weights).fitness

    def tournament(self, population, k=4):
        candidates = self.rng.sample(population, min(k, len(population)))
        return min(candidates, key=self.fitness)

    def crossover(self, p1, p2):
        common = list(set(p1[1:-1]).intersection(p2[1:-1]))
        if not common or self.rng.random() > self.crossover_rate:
            return p1[:], p2[:]

        pivot = self.rng.choice(common)
        i = p1.index(pivot)
        j = p2.index(pivot)

        c1 = self.repair(p1[:i] + p2[j:])
        c2 = self.repair(p2[:j] + p1[i:])

        return c1 or p1[:], c2 or p2[:]

    def mutate(self, path):
        if not path or len(path) <= 2 or self.rng.random() > self.mutation_rate:
            return path

        idx = self.rng.randint(0, len(path) - 2)
        current = path[idx]
        candidates = [n for n in available_neighbors(self.G, current) if n != path[idx + 1]]

        if not candidates:
            return path

        new_node = self.rng.choice(candidates)
        candidate = path[:idx + 1] + [new_node] + path[idx + 1:]
        repaired = self.repair(candidate)

        return repaired or path

    def run(self):
        population = self.initial_population()

        if not population:
            return None, [], []

        history = []
        diversity = []

        for _ in range(self.generations):
            population = [p for p in population if p and p[-1] == self.goal]
            population.sort(key=self.fitness)

            if not population:
                break

            history.append(self.fitness(population[0]))
            diversity.append(len(set(tuple(p) for p in population)))

            new_population = population[:self.elite_size]

            while len(new_population) < self.population_size:
                p1 = self.tournament(population)
                p2 = self.tournament(population)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)

                if c1 and c1[-1] == self.goal:
                    new_population.append(c1)
                if len(new_population) < self.population_size and c2 and c2[-1] == self.goal:
                    new_population.append(c2)

            population = new_population

        population.sort(key=self.fitness)
        return population[0], history, diversity

# -----------------------------
# Visualization helpers
# -----------------------------
def route_dataframe(G, route):
    rows = []
    for i, (u, v) in enumerate(zip(route[:-1], route[1:]), start=1):
        e = G.edges[u, v]
        rows.append({
            "Step": i,
            "From": f"{u}",
            "To": f"{v}",
            "Distance": round(e["distance"], 2),
            "Speed": round(e["speed"], 1),
            "Congestion": round(e["congestion"] * 100, 1),
            "Safety Risk": round(e["safety"] * 100, 1),
            "Blocked": "YES" if e["blocked"] else "NO"
        })
    return pd.DataFrame(rows)

def network_figure(G, route=None):
    fig = go.Figure()

    # edges
    for u, v, d in G.edges(data=True):
        x0, y0 = G.nodes[u]["x"], G.nodes[u]["y"]
        x1, y1 = G.nodes[v]["x"], G.nodes[v]["y"]

        if d["blocked"]:
            width = 2.5
        else:
            width = 1.0 + 3.0 * d["congestion"]

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(width=width, color=("#ef4444" if d["blocked"] else ("#f59e0b" if d["congestion"] > 0.65 else "#3b82f6"))),
            opacity=0.75,
            hoverinfo="text",
            text=f"Congestion: {d['congestion']*100:.1f}%<br>Risk: {d['safety']*100:.1f}%<br>Blocked: {d['blocked']}"
        ))

    if route:
        xs = [G.nodes[n]["x"] for n in route]
        ys = [G.nodes[n]["y"] for n in route]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(width=7, color="#dc2626"),
            marker=dict(size=8, color="#2563eb", line=dict(width=2, color="#ffffff")),
            name="Optimized Evacuation Route"
        ))

    nodes = list(G.nodes)
    fig.add_trace(go.Scatter(
        x=[G.nodes[n]["x"] for n in nodes],
        y=[G.nodes[n]["y"] for n in nodes],
        mode="markers",
        marker=dict(size=5),
        text=[str(n) for n in nodes],
        hovertemplate="Junction %{text}<extra></extra>",
        name="Road Network"
    ))

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False, scaleanchor="x"),
        showlegend=True,
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#34546f")
    )
    return fig

# -----------------------------
# UI
# -----------------------------
st.markdown("""<div class="hero"><div class="status">🔴 EMERGENCY OPERATIONS · LIVE SIMULATION</div><h1>🏥 EVAC-GA Command Center</h1><p><b>AI-Assisted Emergency Evacuation Route Optimization</b></p><p>Find safer evacuation routes by balancing <b>distance, travel time, congestion, hazard risk and blocked roads</b></p></div>""",unsafe_allow_html=True)
st.markdown("""<div class="card"><b>💡 How it works</b><br><span class="muted">The system generates many possible routes, scores them for emergency conditions, evolves better routes through generations, and recommends the best feasible route. It does <b>not</b> simply choose the shortest road.</span></div>""",unsafe_allow_html=True)

with st.sidebar:
    st.header("🎛️ Emergency Control Room")
    st.caption("Change the conditions, then press Optimize Evacuation Route.")
    st.header("⚠️ 1. Emergency Scenario")

    seed = st.number_input("Simulation Seed", min_value=1, max_value=9999, value=42, step=1)

    blockage_rate = st.slider("🚧 Blocked Roads", 0.0, 0.45, 0.12, 0.01)
    st.caption("Chance that a road becomes unavailable.")
    congestion_level = st.slider("🚦 Traffic Congestion", 0.0, 1.0, 0.55, 0.05)
    st.caption("Higher values mean heavier crowding.")
    safety_level = st.slider("⚠️ Hazard Risk", 0.0, 1.0, 0.35, 0.05)
    st.caption("Higher values mean more dangerous conditions.")

    st.divider()
    st.header("🧬 2. Genetic Algorithm")
    st.caption("Controls how extensively the algorithm searches for better routes.")

    population_size = st.slider("Population Size", 30, 180, 80, 10)
    generations = st.slider("Generations", 30, 250, 100, 10)
    mutation_rate = st.slider("Mutation Rate", 0.02, 0.40, 0.15, 0.01)
    crossover_rate = st.slider("Crossover Rate", 0.50, 1.00, 0.85, 0.05)
    elite_size = st.slider("Elite Solutions", 2, 15, 6, 1)

    st.divider()
    st.header("🎯 3. What matters most?")
    st.caption("Higher weight means the optimizer prioritizes that factor more.")

    w_distance = st.slider("Distance", 0.1, 5.0, 1.0, 0.1)
    w_time = st.slider("Travel Time", 0.1, 5.0, 2.0, 0.1)
    w_congestion = st.slider("Congestion", 0.1, 5.0, 2.5, 0.1)
    w_risk = st.slider("Safety Risk", 0.1, 5.0, 3.0, 0.1)

    run = st.button("🚀 OPTIMIZE EVACUATION ROUTE", use_container_width=True)

weights = {
    "distance": w_distance,
    "time": w_time,
    "congestion": w_congestion,
    "risk": w_risk,
    "blocked": 10000
}

if "result" not in st.session_state or run:
    G = build_city(seed=seed)
    apply_scenario(G, blockage_rate, congestion_level, safety_level, seed)

    # Source and candidate exits
    start = (4, 1)
    exits = [(0, 10), (8, 10), (0, 0), (8, 0)]
    available_exits = [e for e in exits if nx.has_path(G, start, e)]

    if not available_exits:
        st.error("No feasible evacuation exit exists under this scenario. Reduce blockage probability.")
        st.stop()

    exit_scores = []
    for goal in available_exits:
        p = nearest_available_path(G, start, goal)
        if p:
            exit_scores.append((route_metrics(G, p, weights).fitness, goal))

    exit_scores.sort()
    goal = exit_scores[0][1]

    ga = EvacuationGA(
        G, start, goal, weights,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        elite_size=elite_size,
        seed=seed
    )

    with st.spinner("Running evolutionary search across evacuation routes..."):
        best_route, history, diversity = ga.run()

    baseline = nearest_available_path(G, start, goal)

    st.session_state.result = {
        "G": G,
        "start": start,
        "goal": goal,
        "route": best_route,
        "baseline": baseline,
        "history": history,
        "diversity": diversity,
        "weights": weights,
        "seed": seed
    }

result = st.session_state.result
G = result["G"]
start = result["start"]
goal = result["goal"]
route = result["route"]
baseline = result["baseline"]
weights = result["weights"]

if route is None:
    st.error("The Genetic Algorithm could not find a feasible route. Lower blockage probability and retry.")
    st.stop()

m_ga = route_metrics(G, route, weights)
m_base = route_metrics(G, baseline, weights)

def pct_improvement(a, b):
    if a == 0:
        return 0
    return (b - a) / b * 100

st.markdown("### 🏁 Recommended Evacuation Plan")
st.caption("These values describe the route selected by the Genetic Algorithm for the current emergency scenario.")
cards=[("📏 Route Distance",f"{m_ga.distance:.2f} km","Total route length"),("⏱️ Estimated Travel",f"{m_ga.time*60:.1f} min","Scenario-based estimate"),("🚦 Congestion",f"{m_ga.congestion*100:.1f}%","Average route congestion"),("🛡️ Safety Risk",f"{m_ga.risk*100:.1f}%","Average modeled risk"),("🧬 Fitness Score",f"{m_ga.fitness:.2f}","Lower is better")]
cols=st.columns(5)
for col,(label,value,note) in zip(cols,cards):
    with col: st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',unsafe_allow_html=True)
st.markdown("""<div class="card"><b>🧭 Route interpretation</b><br><span class="muted">A slightly longer route can be selected when it is safer or less congested. The recommendation is based on the current simulated emergency conditions.</span></div>""",unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Emergency Route Map", "🧬 How GA Optimizes", "📊 Compare Routes", "🔍 Why This Route?", "📥 Export Plan"
])

with tab1:
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.plotly_chart(network_figure(G, route), use_container_width=True)
        st.markdown('<div class="legend"><b>🔵 Road Network</b><b>🟥 Blocked</b><b>🟠 High Congestion</b><b>━━ Recommended Evacuation Route</b></div>',unsafe_allow_html=True)
    with col2:
        st.subheader("Emergency Situation")
        st.write(f"**Source:** {start}")
        st.write(f"**Selected Safe Exit:** {goal}")
        st.write(f"**Roads:** {len(G.edges)}")
        st.write(f"**Blocked Roads:** {sum(1 for _,_,d in G.edges(data=True) if d['blocked'])}")
        st.write(f"**Route Junctions:** {len(route)}")
        st.write(f"**Alternative Exits Evaluated:** {len([(e) for e in [(0,10),(8,10),(0,0),(8,0)] if nx.has_path(G,start,e)])}")
        st.markdown("""<div class="card"><b>🛰️ Emergency route visualization</b><br><span class="muted">This visualization is generated from the simulated emergency road network.</span></div>""",unsafe_allow_html=True)

with tab2:
    st.subheader("🧬 Genetic Algorithm Convergence")
    st.info("Read left to right: falling fitness means the population is finding better routes. A stable curve suggests the search is converging.")

    hist = result["history"]
    div = result["diversity"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(hist)+1)),
        y=hist,
        mode="lines",
        name="Best Fitness"
    ))
    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="Fitness (lower is better)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(range(1, len(div)+1)),
        y=div,
        mode="lines",
        name="Unique Routes"
    ))
    fig2.update_layout(
        xaxis_title="Generation",
        yaxis_title="Population Diversity",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "The population starts with many candidate routes. Selection keeps lower-cost routes, "
        "crossover combines useful route segments, mutation explores alternatives, and elitism "
        "preserves the strongest solutions."
    )

with tab3:
    st.subheader("📊 GA Route vs Shortest-Distance Baseline")
    st.caption("The baseline only minimizes distance. The GA considers distance, time, congestion, safety and blocked roads together.")

    comparison = pd.DataFrame({
        "Metric": ["Distance", "Travel Time", "Congestion", "Safety Risk", "Fitness"],
        "Baseline": [
            m_base.distance,
            m_base.time,
            m_base.congestion * 100,
            m_base.risk * 100,
            m_base.fitness
        ],
        "Genetic Algorithm": [
            m_ga.distance,
            m_ga.time,
            m_ga.congestion * 100,
            m_ga.risk * 100,
            m_ga.fitness
        ]
    })

    st.dataframe(comparison.round(3), use_container_width=True, hide_index=True)

    improvements = pd.DataFrame({
        "Metric": ["Distance", "Travel Time", "Congestion", "Safety Risk", "Overall Fitness"],
        "Improvement (%)": [
            pct_improvement(m_ga.distance, m_base.distance),
            pct_improvement(m_ga.time, m_base.time),
            pct_improvement(m_ga.congestion, m_base.congestion),
            pct_improvement(m_ga.risk, m_base.risk),
            pct_improvement(m_ga.fitness, m_base.fitness)
        ]
    })

    st.bar_chart(improvements.set_index("Metric"))

    st.subheader("Optimized Route Details")
    st.dataframe(route_dataframe(G, route), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Why did the Genetic Algorithm choose this route?")

    contributions = {
        "Distance": weights["distance"] * m_ga.distance,
        "Travel Time": weights["time"] * m_ga.time,
        "Congestion": weights["congestion"] * m_ga.crowd_exposure,
        "Safety Risk": weights["risk"] * m_ga.risk,
        "Blocked Road Penalty": weights["blocked"] * m_ga.blocked_edges
    }

    explain = pd.DataFrame({
        "Objective": list(contributions.keys()),
        "Contribution": list(contributions.values())
    }).sort_values("Contribution", ascending=False)

    fig = go.Figure(go.Bar(
        x=explain["Contribution"],
        y=explain["Objective"],
        orientation="h"
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        **Decision logic:** The optimizer did not simply minimize distance.
        It evaluated every candidate route using a weighted multi-objective fitness function.

        **Distance weight:** {weights['distance']:.1f}  
        **Time weight:** {weights['time']:.1f}  
        **Congestion weight:** {weights['congestion']:.1f}  
        **Safety weight:** {weights['risk']:.1f}

        The result is a route designed for the emergency scenario rather than a normal-day shortest path.
        """
    )

with tab5:
    st.subheader("Export optimized evacuation plan")

    export_df = route_dataframe(G, route)
    csv = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Route CSV",
        data=csv,
        file_name="optimized_evacuation_route.csv",
        mime="text/csv"
    )

    summary = {
        "simulation_seed": int(result["seed"]),
        "source": str(start),
        "safe_exit": str(goal),
        "route_nodes": len(route),
        "distance_km": round(m_ga.distance, 3),
        "travel_time_minutes": round(m_ga.time * 60, 3),
        "average_congestion_percent": round(m_ga.congestion * 100, 3),
        "average_safety_risk_percent": round(m_ga.risk * 100, 3),
        "fitness": round(m_ga.fitness, 3)
    }

    st.json(summary)

st.divider()
st.markdown("""<div class="card"><h3>🧠 Project Methodology</h3><span class="muted"><b>1. Model:</b> represent the city as a graph. &nbsp; <b>2. Simulate:</b> add congestion, risk and blocked roads. &nbsp; <b>3. Generate:</b> create candidate routes. &nbsp; <b>4. Evaluate:</b> calculate multi-objective fitness. &nbsp; <b>5. Evolve:</b> selection + crossover + mutation + elitism. &nbsp; <b>6. Recommend:</b> choose the lowest-fitness feasible route.</span></div>""",unsafe_allow_html=True)

st.divider()
st.caption("EVAC-GA is a simulation/academic prototype. Real emergency deployment requires validated GIS, live traffic, verified road closures, official evacuation plans and safety authorities.")
