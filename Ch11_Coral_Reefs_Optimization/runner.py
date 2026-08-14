import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import opfunu

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cro import CRO

def run_cro(dim=10, pop_size=50, max_iter=30):
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    func_classes = [getattr(opfunu.cec_based.cec2022, f"F{i}2022") for i in range(1, 13)]
    
    plt.figure(figsize=(12, 8))
    plt.title('CRO Convergence on CEC 2022 Functions (30 Iterations)', fontsize=16)
    
    for func_class in func_classes:
        func_name = func_class.__name__
        print(f"Executing CRO on {func_name}...")
        
        func_instance = func_class(ndim=dim)
        obj_func = lambda x: func_instance.evaluate(x)
        
        lb, ub = func_instance.bounds[0], func_instance.bounds[1]
        b = (lb[0], ub[0]) if isinstance(lb, (list, np.ndarray)) else (lb, ub)
        
        alg = CRO(objective_func=obj_func, bounds=b, dim=dim, pop_size=pop_size, max_iter=max_iter)
        _, _ = alg.optimize()
        
        curve = np.array(alg.convergence_curve)
        curve = np.maximum(curve, 1e-10) 
        plt.plot(curve, label=func_name, linewidth=2)
            
    plt.xlabel('Iterations')
    plt.ylabel('Fitness (Log Scale)')
    plt.yscale('log')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'convergence_plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Execution complete. Plot saved at {plot_path}")

if __name__ == "__main__":
    run_cro()
