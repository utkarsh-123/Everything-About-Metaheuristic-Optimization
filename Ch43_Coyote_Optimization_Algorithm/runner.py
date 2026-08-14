import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import opfunu

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from coa import COA

def run_coa(dim=10, pop_size=50, max_iter=30):
    output_dir = os.path.dirname(os.path.abspath(__file__))
    func_classes = [getattr(opfunu.cec_based.cec2022, f"F{i}2022") for i in range(1, 13)]
    
    plt.figure(figsize=(12, 8))
    plt.title('COA Convergence on CEC 2022 (30 Iterations)', fontsize=16)
    
    for func_class in func_classes:
        func_name = func_class.__name__
        func_instance = func_class(ndim=dim)
        obj_func = lambda x: func_instance.evaluate(x)
        lb, ub = func_instance.bounds[0], func_instance.bounds[1]
        b = (lb[0], ub[0]) if isinstance(lb, (list, np.ndarray)) else (lb, ub)
        
        alg = COA(objective_func=obj_func, bounds=b, dim=dim, pop_size=pop_size, max_iter=max_iter)
        alg.optimize()
        curve = np.maximum(alg.convergence_curve, 1e-10) 
        plt.plot(curve, label=func_name, linewidth=2)
            
    plt.xlabel('Iterations'); plt.ylabel('Fitness (Log)'); plt.yscale('log')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1)); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_plot.png'))
    plt.close()

if __name__ == "__main__": run_coa()
