from dwdynamics import ComplexDynamicsProblem, Objective, helpers 
import json
import os
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BQM, BINARY
import pandas as pd
from io import StringIO
import numpy as np
import subprocess

class Instance:
    def __init__(
            self,
            instance_id: int
    ):
        self._id = instance_id
        self.basepath ='../' if os.getcwd()[-9:] == 'notebooks' else '' # for execution in jupyter notebooks

    
    def create_instance(self, precision: int, number_time_points:int, save = False):
        """
            Creates an instance based on the provided parameters and saves the qubo to /data/instances
        """
        self.precision = precision
        self.number_time_points = number_time_points
        instance_dict = helpers.get_instance(self._id)
        self.H = instance_dict['H']
        self.psi0 = instance_dict['psi0']

        self.problem = ComplexDynamicsProblem(
            hamiltonian= self.H, 
            initial_state = self.psi0,            
            times=tuple(range(number_time_points)),             
            num_bits_per_var=precision                
        )
        self.qubo = self.problem.qubo(objective=Objective.norm)           
        assert self.qubo.num_variables == self.problem.hamiltonian.shape[0] * len(self.problem.times) * self.problem.num_bits_per_var * 2

        # save instances in the form 
        # systemid_{d}_precision_{d}_timepoints_{d}.json
        if save:
            path = f"data/instances/pruned/{self._id}"

            file_name = os.path.join(self.basepath, path, f"precision_{precision}_timepoints_{number_time_points}.json")
            os.makedirs(path, exist_ok=True)
            with open(file_name,'w') as f:
                json.dump(self.qubo.to_serializable(),f)


    def get_sampleset(self,solver_id="5.4"):
        """
            Runs one sample on the indicated D-Wave machine
        """
        if solver_id == "5.4":
            dw_sampler = EmbeddingComposite(DWaveSampler( solver="Advantage_system5.4", region="eu-central-1", ))
        elif solver_id == "2.6": # zephyr
            dw_sampler = EmbeddingComposite(DWaveSampler( solver="Advantage2_prototype2.6"))
        elif solver_id == "6.4": 
            dw_sampler = EmbeddingComposite(DWaveSampler(solver="Advantage_system6.4"))
        else:
            raise ValueError("Invalid solver id")

        self.dw_result = dw_sampler.sample(self.problem.qubo, num_reads=1000, annealing_time=200)

        path = f"data/results/pruned/{self._id}/{solver_id}"
        path = os.path.join(self.basepath, path)
        os.makedirs(path, exist_ok=True)
        idx = helpers.get_last_index(os.listdir(path)) +1

        file_name = os.path.join(path, f"precision_{self.precision}_timepoints_{self.number_time_points}_{idx}.json")

        with open(file_name,'w') as f:
            json.dump(self.dw_result.to_serializable(),f)
    
    def to_xubo(self):
        """
            Convert a BQM to a xubo readable file
        """
        bqm = self.qubo.spin
        # map linear terms
        lin_map = {}
        for i,key in enumerate(sorted(bqm.linear)):
            lin_map[key] = i
        
        output = [f'# QUBITS {len(bqm.linear)}\n']
        output += [f'# offset {bqm.offset}\n']
        output += [f'# quibitmap {lin_map}\n']
        output += [f'# vartype {bqm.vartype}\n']
        output += [f"{lin_map[k]} {lin_map[k]} {v}\n" for k, v in sorted(bqm.linear.items())]
        output += [f"{lin_map[k[0]]} {lin_map[k[1]]} {v}\n" for k, v in bqm.quadratic.items()]
        print(os.getcwd())
        path = f'data/xubo/ising/{self._id}/'
        os.makedirs(path,exist_ok=True)

        with open(os.path.join(self.basepath,path,f'precision_{self.precision}_timepoints_{self.number_time_points}.ising'), 'w') as f:
            f.writelines(output)
            f.close()   

        script_path = os.path.join(self.basepath, 'scripts/run_xubo.sh')
        subprocess.check_call(
            f"{script_path} %s %s %s" % (str(self._id), str(self.precision), str(self.number_time_points)),
            shell=True
)
        

    def get_xubo_df(self)->pd.DataFrame:
        filename = f'precision_{self.precision}_timepoints_{self.number_time_points}.xubo'
        path = os.path.join(self.basepath, f'data/xubo/output/{self._id}', filename)
        if not os.path.exists(path):
            print("running xubo")
            self.to_xubo()

        content = ""
        i = 1
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                line = f.readline()
                if i >= 17:
                    content += line
                i+=1

        assert content[0:6] == 'Energy', 'File does not have correct structure'
        return pd.read_csv(StringIO(content),sep=r'\s+',dtype={'Energy':np.float64,'State':'str'})

    def get_qubo(self) -> BQM:
        return self.qubo




