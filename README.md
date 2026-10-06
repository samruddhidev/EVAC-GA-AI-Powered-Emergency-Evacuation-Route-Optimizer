# 🚨 EVAC-GA — Emergency Evacuation Route Optimizer

**EVAC-GA (Emergency Evacuation using Genetic Algorithm)** is an AI-assisted emergency route optimization system built with **Python, Streamlit, NetworkX, NumPy, Pandas and Plotly**.

The system simulates an emergency road network with **blocked roads, traffic congestion and safety hazards**, then uses a **Genetic Algorithm (GA)** to find an evacuation route that balances distance, travel time, congestion and risk.

> ⚠️ **Academic Prototype:** EVAC-GA is a simulation-based project and is not intended for real-world emergency deployment without validated GIS data, live traffic information, verified road closures and approval from relevant authorities.

---

## 🌟 Features

* 🗺️ Simulated city road network using NetworkX
* 🚧 Dynamic road blockage simulation
* 🚦 Traffic congestion modeling
* ⚠️ Safety/hazard risk modeling
* 🧬 Genetic Algorithm-based route optimization
* 📏 Distance-aware route selection
* ⏱️ Travel-time estimation
* 🛡️ Safety-risk evaluation
* 🔄 Selection, crossover, mutation and elitism
* 📊 GA convergence visualization
* 📈 Route comparison with shortest-distance baseline
* 🔍 Explanation of why a route was selected
* 📥 Export optimized route as CSV
* 🎛️ Interactive Streamlit controls
* 📱 Interactive Plotly route visualization

---

## 🧠 How It Works

EVAC-GA represents the city as a **graph**:

* **Nodes** → road junctions
* **Edges** → roads connecting junctions
* **Edge attributes** → distance, speed, congestion, safety risk, capacity and blockage status

The system then creates an emergency scenario by modifying the road network according to the selected blockage, congestion and safety levels.

### Optimization Pipeline

```text
Emergency Scenario
        ↓
Generate Road Network
        ↓
Apply Blockages & Hazards
        ↓
Generate Candidate Routes
        ↓
Calculate Route Fitness
        ↓
Selection
        ↓
Crossover
        ↓
Mutation
        ↓
Elitism
        ↓
Repeat for Multiple Generations
        ↓
Best Feasible Evacuation Route
```

---

## 🧬 Genetic Algorithm

The Genetic Algorithm searches through multiple possible evacuation routes instead of simply selecting the shortest path.

Each candidate route is evaluated using a weighted fitness function:

```text
Fitness =
    Distance Weight × Distance
  + Time Weight × Travel Time
  + Congestion Weight × Crowd Exposure
  + Risk Weight × Safety Risk
  + Blocked Road Penalty × Blocked Edges
```

### Genetic Operations

**1. Population Initialization**

Multiple feasible routes are generated between the emergency source and evacuation exits.

**2. Selection**

Tournament selection chooses stronger candidate routes based on their fitness.

**3. Crossover**

Useful sections from two parent routes are combined to produce new candidate routes.

**4. Mutation**

Small changes are introduced into routes to explore alternative paths.

**5. Elitism**

The best-performing routes are preserved between generations.

**6. Repair**

Invalid or broken routes are repaired using feasible shortest subpaths.

---

## 🎯 Multi-Objective Optimization

Unlike a conventional shortest-path algorithm, EVAC-GA considers multiple emergency conditions simultaneously.

| Factor          | Purpose                          |
| --------------- | -------------------------------- |
| 📏 Distance     | Minimize total route length      |
| ⏱️ Travel Time  | Reduce estimated evacuation time |
| 🚦 Congestion   | Avoid heavily crowded roads      |
| 🛡️ Safety Risk | Prefer safer roads               |
| 🚧 Blockages    | Avoid unavailable roads          |

This allows the system to select a **slightly longer route if it is significantly safer or less congested**.

---

## 🖥️ Application Dashboard

The Streamlit application contains five main sections:

### 🗺️ 1. Emergency Route Map

Displays:

* Simulated road network
* Blocked roads
* Congested roads
* Optimized evacuation route
* Source location
* Selected safe exit

### 🧬 2. How GA Optimizes

Visualizes:

* Best fitness across generations
* Population diversity

This helps demonstrate whether the Genetic Algorithm is converging toward better solutions.

### 📊 3. Compare Routes

Compares the Genetic Algorithm route against a shortest-distance baseline using:

* Distance
* Travel time
* Congestion
* Safety risk
* Overall fitness

### 🔍 4. Why This Route?

Shows the contribution of different objectives to the final route fitness.

This provides an interpretable explanation of the optimizer's decision.

### 📥 5. Export Plan

Allows users to download the optimized route as:

```text
optimized_evacuation_route.csv
```

The application also displays a JSON summary containing the main evacuation metrics.

---

## 🎛️ Interactive Parameters

The sidebar allows the user to modify the emergency simulation.

### Emergency Scenario

* Simulation Seed
* Blocked Roads
* Traffic Congestion
* Hazard Risk

### Genetic Algorithm

* Population Size
* Number of Generations
* Mutation Rate
* Crossover Rate
* Elite Solutions

### Optimization Weights

* Distance
* Travel Time
* Congestion
* Safety Risk

This makes it possible to experiment with different emergency scenarios and optimization priorities.

---

## 🛠️ Tech Stack

| Technology            | Purpose                            |
| --------------------- | ---------------------------------- |
| **Python**            | Core programming language          |
| **Streamlit**         | Interactive web application        |
| **NetworkX**          | Graph and road-network modeling    |
| **NumPy**             | Numerical operations               |
| **Pandas**            | Data processing and route analysis |
| **Plotly**            | Interactive visualizations         |
| **Genetic Algorithm** | Route optimization                 |

---

## 📁 Project Structure

```text
EVAC-GA/
│
├── app.py
├── requirements.txt
├── README.md
└── optimized_evacuation_route.csv
```

> `optimized_evacuation_route.csv` is generated by the application and does not need to be committed to the repository unless you want to include an example output.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/EVAC-GA.git
cd EVAC-GA
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
streamlit
numpy
pandas
networkx
plotly
```

---

## 💡 Example Scenario

Suppose an emergency occurs in the simulated city.

Several roads may become:

* Blocked 🚧
* Highly congested 🚦
* High-risk ⚠️

A traditional shortest-path algorithm may select a short route that passes through a highly congested or risky area.

EVAC-GA evaluates multiple candidate routes and can instead select a route that is:

> **slightly longer but faster, safer and more feasible under the current emergency conditions.**

---

## 📊 Output Metrics

The dashboard provides the following key metrics for the selected route:

* **Route Distance**
* **Estimated Travel Time**
* **Average Congestion**
* **Average Safety Risk**
* **Fitness Score**
* **Number of Route Junctions**
* **Blocked Roads**
* **Alternative Exits Evaluated**

A lower fitness score represents a better route according to the selected optimization weights.

---

## 🔬 Project Methodology

```text
1. Model
   ↓
Represent the city as a graph.

2. Simulate
   ↓
Introduce congestion, safety risks and blocked roads.

3. Generate
   ↓
Create feasible candidate evacuation routes.

4. Evaluate
   ↓
Calculate multi-objective route fitness.

5. Evolve
   ↓
Apply selection, crossover, mutation and elitism.

6. Recommend
   ↓
Select the lowest-fitness feasible route.
```

---

## 🎓 Applications

EVAC-GA demonstrates how evolutionary optimization can be applied to:

* Emergency evacuation planning
* Disaster management simulations
* Traffic-aware routing
* Hospital evacuation scenarios
* Smart-city simulations
* Risk-aware navigation
* AI-based decision support systems

---

## 🔮 Future Scope

The current system uses a simulated road network. Future versions could integrate:

* 🗺️ Real GIS/map data
* 🚗 Real-time traffic APIs
* 🚧 Live road closure information
* 🌦️ Weather and disaster data
* 🏥 Real hospital and shelter locations
* 👥 Population-density information
* 🤖 Reinforcement Learning for adaptive routing
* 📡 IoT-based emergency sensors
* 📍 GPS-based evacuation guidance
* ☁️ Cloud deployment
* 🧠 Machine Learning-based risk prediction

---

## ⚠️ Limitations

The current implementation is a simulation and has several limitations:

* Road networks are randomly generated.
* Traffic and safety values are simulated.
* Road blockage is probabilistic.
* Travel-time calculations are scenario-based.
* The system does not use live GPS or traffic data.
* The model has not been validated for real emergency operations.

Therefore, the application should be considered an **academic/research prototype rather than an operational emergency management system**.

---

## 👩‍💻 Author

**Samruddhi Jain**

B.Tech Artificial Intelligence & Data Science
K.J. Somaiya School of Engineering

---

## ⭐ If you found this project interesting

Feel free to **star ⭐ the repository**, explore the implementation and experiment with different emergency scenarios and Genetic Algorithm parameters.

---

## 📜 License

This project is intended for **educational and academic purposes**.
