# Metaheuristic Optimization Algorithms — Code Repository

This repository contains **53 metaheuristic optimization algorithms** implemented **entirely from scratch in Python**, as presented in the book *Everything About Optimization*. Each algorithm is benchmarked on the **CEC 2022 benchmark suite** (12 functions) and produces a convergence plot.

---

## 📁 Repository Structure

Each algorithm lives in its own folder named `ChXX_AlgorithmName/` and contains:

| File | Description |
|------|-------------|
| `[algorithm].py` | Core algorithm implementation (no third-party optimisation libraries) |
| `runner.py` | Executes the algorithm on all 12 CEC 2022 functions and saves the convergence plot |
| `convergence_plot.png` | Generated after running `runner.py` |

A shared `utils/base_algorithm.py` provides the common `BaseAlgorithm` interface used by all algorithms.

---

## ⚙️ Requirements

Install the required Python packages:

```bash
pip install numpy matplotlib opfunu
```

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computation |
| `matplotlib` | Convergence plot generation |
| `opfunu` | CEC 2022 benchmark function suite |

---

## 🚀 Running an Algorithm

Navigate to any algorithm folder and run its `runner.py`:

```bash
cd "Ch02_Cuckoo_Search"
python runner.py
```

This will:
1. Execute the algorithm on all **12 CEC 2022 benchmark functions**
2. Run **30 iterations** per function
3. Save a `convergence_plot.png` in the same folder

### Customising Parameters

Open `runner.py` and modify the function call arguments:

```python
run_cs(
    dim=10,        # Problem dimensionality (10, 20, 30, 50, 100)
    pop_size=50,   # Population size
    max_iter=30    # Number of iterations
)
```

---

## 📊 Benchmark Configuration

| Setting | Value |
|---------|-------|
| Benchmark Suite | CEC 2022 (F1–F12) |
| Search Bounds | [-100, 100]^D |
| Dimensionality | D = 10 (default) |
| Iterations | 30 |
| Plot | Fitness (log scale) vs. Iterations |

---

## 📚 Algorithm Index

### Part I — Nature-Inspired Swarm Algorithms

| Folder | Algorithm | Abbreviation | Inspired By |
|--------|-----------|--------------|-------------|
| `Ch02_Cuckoo_Search` | Cuckoo Search | CS | Cuckoo bird brood parasitism + Lévy flights |
| `Ch03_Lion_Optimization_Algorithm` | Lion Optimization Algorithm | LOA | Pride and nomad lion hunting behaviour |
| `Ch04_Aquila_Optimization` | Aquila Optimization | AO | Hunting tactics of Aquila eagle |
| `Ch05_Hunger_Games_Search` | Hunger Games Search | HGS | Hunger-driven animal foraging |
| `Ch06_Komodo_Mlipir_Algorithm` | Komodo Mlipir Algorithm | KMA | Komodo dragon territory and hunting |
| `Ch07_Flying_Foxes_Optimization` | Flying Foxes Optimization | FFO | Roosting and foraging of flying foxes |
| `Ch08_Drone_Squadron_Optimization` | Drone Squadron Optimization | DSO | Military drone swarm tactics |
| `Ch09_Color_Harmony_Algorithm` | Color Harmony Algorithm | CHA | Aesthetic colour harmony perception |
| `Ch10_Flower_Pollination_Algorithm` | Flower Pollination Algorithm | FPA | Flower pollination via insects and wind |
| `Ch11_Coral_Reefs_Optimization` | Coral Reefs Optimization | CRO | Coral reef broadcast spawning and settlement |
| `Ch12_Slime_Mould_Algorithm` | Slime Mould Algorithm | SMA | Oscillatory foraging of slime mould |
| `Ch13_Ideal_Gas_Molecular_Movement` | Ideal Gas Molecular Movement | IGMM | Kinetic theory of ideal gas molecules |
| `Ch41_Grasshopper_Optimization_Algorithm` | Grasshopper Optimization Algorithm | GOA | Swarm behaviour of grasshoppers |
| `Ch42_Butterfly_Optimization_Algorithm` | Butterfly Optimization Algorithm | BOA | Fragrance-guided foraging of butterflies |
| `Ch43_Coyote_Optimization_Algorithm` | Coyote Optimization Algorithm | COA | Pack social hierarchy of coyotes |
| `Ch44_Emperor_Penguins_Colony` | Emperor Penguins Colony | EPC | Huddle thermoregulation behaviour |
| `Ch45_Emperor_Penguin_Optimizer` | Emperor Penguin Optimizer | EPO | Temperature-phased huddling with boundaries |
| `Ch46_Bonobo_Optimizer` | Bonobo Optimizer | BO | Mating strategies of Bonobos |
| `Ch47_Fitness_Dependent_Optimizer` | Fitness Dependent Optimizer | FDO | Fitness-weighted swarming of bees |
| `Ch48_Chimp_Optimization_Algorithm` | Chimp Optimization Algorithm | ChOA | Four social roles of chimpanzees |
| `Ch49_Mayfly_Algorithm` | Mayfly Algorithm | MA | Mating flight of mayflies |
| `Ch50_African_Vultures_Optimization_Algorithm` | African Vultures Optimization Algorithm | AVOA | Foraging & navigation of African vultures |
| `Ch51_Coot_Optimization_Algorithm` | Coot Optimization Algorithm | CootOA | Swimming chain behaviour of coots |
| `Ch52_Golden_Eagle_Optimizer` | Golden Eagle Optimizer | GEO | Attack and cruise flight of golden eagles |
| `Ch38_Particle_Swarm_Optimization` | Particle Swarm Optimization | PSO | Social behaviour of bird flocks/fish schools |
| `Ch39_Artificial_Bee_Colony` | Artificial Bee Colony | ABC | Foraging behaviour of honey bees |
| `Ch40_Firefly_Algorithm` | Firefly Algorithm | FA | Light-attraction of fireflies |
| `Ch37_Ant_Colony_Optimization` | Ant Colony Optimization | ACO | Pheromone trail of ant colonies |

---

### Part II — Physics & Chemistry Inspired

| Folder | Algorithm | Abbreviation | Inspired By |
|--------|-----------|--------------|-------------|
| `Ch14_Henry_Gas_Solubility_Optimization` | Henry Gas Solubility Optimization | HGSO | Henry's law of gas solubility |
| `Ch15_Equilibrium_Optimizer` | Equilibrium Optimizer | EO | Control-volume mass-balance equilibrium |
| `Ch31_Gravitational_Search_Algorithm` | Gravitational Search Algorithm | GSA | Newtonian gravity and laws of motion |
| `Ch32_Water_Cycle_Algorithm` | Water Cycle Algorithm | WCA | Hydrological water cycle |
| `Ch33_Lightning_Search_Algorithm` | Lightning Search Algorithm | LSA | Propagation of lightning via projectiles |
| `Ch34_Thermal_Exchange_Optimization` | Thermal Exchange Optimization | TEO | Newton's law of cooling |
| `Ch30_Golden_Ratio_Optimization` | Golden Ratio Optimization | GRO | The golden ratio (φ = 1.618) |
| `Ch28_Heap_based_Optimizer` | Heap-based Optimizer | HBO | Corporate heap hierarchy |
| `Ch29_Sine_Cosine_Algorithm` | Sine Cosine Algorithm | SCA | Sine and cosine mathematical functions |

---

### Part III — Human & Social Behaviour Inspired

| Folder | Algorithm | Abbreviation | Inspired By |
|--------|-----------|--------------|-------------|
| `Ch16_Genetic_Algorithms` | Genetic Algorithms | GA | Natural selection and genetics |
| `Ch17_Cultural_Algorithm` | Cultural Algorithm | CA | Cultural evolution and belief spaces |
| `Ch18_Differential_Evolution` | Differential Evolution | DE | Differential vectors among population |
| `Ch19_Harmony_Search_Algorithm` | Harmony Search Algorithm | HSA | Jazz musical improvisation |
| `Ch20_Teaching_Learning_Based_Optimization` | Teaching-Learning-Based Optimization | TLBO | Classroom teacher-learner interaction |
| `Ch21_Human_Behavior_based_Optimization` | Human Behavior-based Optimization | HBBO | Career and professional field dynamics |
| `Ch22_Ideology_Algorithm` | Ideology Algorithm | IA | Political party competition |
| `Ch23_Farmland_Fertility` | Farmland Fertility | FF | Agricultural soil fertility management |
| `Ch24_Future_Search_Algorithm` | Future Search Algorithm | FSA | Human search for a better life |
| `Ch25_Coronavirus_Herd_Immunity_Optimizer` | Coronavirus Herd Immunity Optimizer | CHIO | COVID-19 infection and immunity dynamics |
| `Ch26_Giza_Pyramids_Construction` | Giza Pyramids Construction | GPC | Physics of ramp construction in ancient Egypt |
| `Ch27_Group_Teaching_Optimization_Algorithm` | Group Teaching Optimization Algorithm | GTOA | Outstanding vs. average student groups |
| `Ch35_Volleyball_Premier_League` | Volleyball Premier League | VPL | Volleyball league competition and player trades |
| `Ch36_Ludo_Game_based_Swarm_Intelligence` | Ludo Game-based Swarm Intelligence | LGSI | Dice roll outcomes in Ludo game |
| `Ch53_Simulated_Annealing` | Simulated Annealing | SA | Annealing process in metallurgy |
| `Ch54_Tabu_Search` | Tabu Search | TS | Memory-based prohibition of revisits |

---

## 🔧 Utility: Base Algorithm

All algorithms inherit from `utils/base_algorithm.py`:

```python
from utils.base_algorithm import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        ...
```

**Provided methods:**

| Method | Description |
|--------|-------------|
| `initialize_population()` | Uniform random initialisation within `[lb, ub]^D` |
| `evaluate(population)` | Vectorised fitness evaluation |
| `enforce_bounds(population)` | Clips all positions to `[lb, ub]` |
| `update_global_best(population, fitness)` | Updates `global_best_pos` and `global_best_score` |
| `convergence_curve` | List recording best fitness at each iteration |

---

## 📈 Convergence Plot

Each `runner.py` generates a single plot (`convergence_plot.png`) containing all 12 CEC 2022 function curves:

- **X-axis**: Iteration number (0 to `max_iter`)
- **Y-axis**: Best fitness found (log scale)
- **12 coloured lines**: One per CEC 2022 function (F1–F12)

---

## 📝 Notes

- **No third-party optimisation libraries** are used. All algorithm logic is implemented from scratch.
- **Simulated Annealing (Ch53)** and **Tabu Search (Ch54)** are single-solution (trajectory-based) methods — `pop_size` is not applicable.
- **Mayfly Algorithm (Ch49)** uses a split male/female population; effective population is `pop_size // 2` per gender.
- CEC 2022 functions can shift their optima; results will differ between runs unless seeds are fixed.

---

## 🗂️ Quick Reference — Running All Algorithms

To run all algorithms sequentially, execute from the `Code Examples` directory:

```powershell
Get-ChildItem -Directory | ForEach-Object { python "$($_.Name)\runner.py" }
```

> ⚠️ This will take considerable time. Run individual algorithms for quicker results.
