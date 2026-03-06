# Physical Learning (JAX) — Continual Learning in a Resistor Network

A small, **reproducible JAX** demo inspired by “physical learning”: a physical network adapts through **local parameter updates** (edge conductances) and solves tasks via **Kirchhoff/Laplacian physics**, without digital neurons.

This repo is designed to be a clean, readable computational signal for theoretical/computational work on **learning in physical systems**.

---

## What this demonstrates (mapped to the AMOLF framing)

### Physical learning + energy landscape
- For fixed boundary inputs, node potentials are obtained by solving a **Dirichlet problem** on a weighted graph Laplacian.
- The resulting state is the minimiser of a quadratic **energy**:
  \[
  E(v; w) = \frac{1}{2}\sum_{(i,j)\in \mathcal{E}} w_{ij}(v_i - v_j)^2.
  \]
- “Learning” reshapes the effective energy landscape through changes in conductances \(w\).

### Learning modes / expressiveness (linear baseline)
- This is a **linear physical network** baseline. It has restricted expressiveness (compared to nonlinear networks), but is ideal for:
  - defining **learning signals** cleanly,
  - understanding task imprinting in parameters,
  - building intuition before adding nonlinear elements (diodes, saturating flows, piecewise conductances).

### Continual learning under physical constraints
- Train the network on **Task A**, then train on **Task B**.
- Observe **catastrophic forgetting**: Task A performance degrades after learning Task B.
- Add a simple **stability penalty** (anchoring parameters to Task A solution) to reduce forgetting, revealing the classic **plasticity–stability trade-off**.

---

## Repository structure
physical-learning-continual/
README.md
requirements.txt
src/
network.py
train_single.py
train_continual.py
make_figures.py
notebooks/
demo.ipynb # optional (can be added later)
fig_forgetting.png
fig_learning_signal.png
LICENSE


---

## Quickstart

### 1) Create environment + install dependencies
```bash
python -m venv .venv
# activate:
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -u src/train_continual.py

## Results

**Forgetting after learning Task B**  
![Forgetting on Task A after learning Task B](fig_forgetting.png)

**Learning signal (parameter change)**  
![Learning signal: absolute parameter change after Task B](fig_learning_signal.png)

---

## Notes on the model

- **Network:** an undirected graph with positive conductances \(w\) on edges.  
- **State solve:** solve \(L(w)\,v = 0\) on free nodes with Dirichlet boundary conditions on input nodes.  
- **Training:** optimise raw parameters \(w_{\text{raw}}\) using JAX autodiff, mapped to \(w>0\) via softplus:
  \[
  w = \mathrm{softplus}(w_{\text{raw}}) + \epsilon .
  \]
- **Tasks:** supervised mappings from boundary inputs to readout node potentials.  
- **Stability penalty:** quadratic penalty toward the Task A parameters (simple anchor/EWC-style baseline).

---

## How to interpret the figures

- **Learning signal:** \(\lvert \Delta w \rvert\) after Task B highlights which edges “move” to encode the new task; it’s a minimal proxy for a **physical imprint** of learning.  
- **Forgetting:** the increase in Task A error after training Task B quantifies interference; the stabilised run reduces forgetting at the cost of Task B performance.

---

## Suggested next extensions (if continuing)

- **Nonlinear physical elements:** diode-like or saturating conductances to study nonlinear learning modes.  
- **Capacity / expressiveness:** sweep over task families and quantify how many tasks can be learned before interference dominates.  
- **More principled continual learning:** curvature-aware penalties, constraint-based updates, or structure-aware regularisation.  
- **Task imprint analysis:** relate \(\Delta w\) patterns to task structure (e.g., which input combinations drive which subgraphs).

---

## Reproducibility

- Small dependency footprint (JAX + NumPy + Matplotlib).  
- Scripts are deterministic given the fixed PRNG seeds in the training scripts.

---

## License

MIT .