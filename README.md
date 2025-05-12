
# 🔁 Spectral Logic Memory Simulation (谱计算记忆模拟)

This repository implements a physically inspired simulation of **programmable spectral computing**, following the theoretical framework of modal slot logic, entropy feedback, and noncommutative kernel interactions as described in advanced spectral computing models.

## 🔬 Features

✅ Frequency-slot based memory activation and freezing  
✅ Entropy H(t) calculation and feedback control  
✅ Interference using a noncommutative kernel \( K(z, z') \)  
✅ Dynamic optimization via structure energy function \( L[\phi] \)  
✅ Noise-injected modal environment for robustness testing  
✅ YAML-style modal scheduling

## 📈 Output

Running the main script `p.py` will produce a 4-panel graph:

1. **ϕ(t)**: Combined modal output signal  
2. **S(t)**: Modal control signals per slot  
3. **H(t), IF(t)**: Structural entropy and modal variance rate  
4. **L[ϕ]**: The spectral optimization function (entropy + interference + modal variance)



> Example snapshot from `p.py` execution:
> - Spectral slot 0: Active → Frozen  
> - Spectral slot 1 & 3: Staggered activation  
> - Frozen slots retain exact modal value (memory behavior)  
> - Entropy feedback suppresses unstable modal behavior  
> - Optimization function L[ϕ] reflects system tension

https://zenodo.org/records/15384932
