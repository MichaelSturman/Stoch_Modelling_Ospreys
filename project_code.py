import pandas as pd
import networkx as nx
import re
import matplotlib.pyplot as plt
import numpy as np
import random
from numpy.random import Generator, PCG64

# Project - ecosystem modelling using coupled poisson processes.
# https://en.wikipedia.org/wiki/Food_chain#/media/File:Chesapeake_Waterbird_Food_Web.jpg
# Let's go with Ospreys (in ocean city maybe)

# https://conservewildlifenj.org/wp-content/uploads/2026/05/2025-NJ-Osprey-Project-Report.pdf
# 52 ospreys in 2025. Run with that population.
# Arrival time - April

# https://birdsoftheworld.org/bow/species/osprey/cur/introduction


# Biomass modelling for fish / ocean health?
#   - function of temperature
#   - function of ocean PH? (human / environmentally influenced)
# O
# Biomass of fish reduced by human fishing at a constant rate
# Non-homogenous poisson process for osprey feeding - related 



class eco_sim_basic():
    def __init__(self, osprey_pairs: int, run_time: int, nruns: int, master_seed: int):

        self.t = 0
        self.run_time = run_time
        self.nruns = nruns
        self.master_seed = master_seed
        
        # count of osprey pairs
        self.osprey_pairs = osprey_pairs
        self.eco_graph = nx.Graph()

        self.data_out = pd.DataFrame()

        # egg laying rate
        self.eggless_pairs = []
        self.lam_lay = 1 # to be updated using function defined below
        self.lam_hatch = 0
        self.lam_death = 0
        self.lam_fledge = 0

        self.survival_prop = 0.95 # chick survival proportion - "C:\Villanova\year_2\semester_1\stoch_modeling\Project\Papers\fmars-10-1284462.pdf" 
                                  # use this?
                                  # add in sibling order / hierarchy?
        
        self.eggs = []
        self.hatched = []
        self.failed = []
        self.fledged = []
        self.deceased = [] # deceased includes unhatched eggs, deceased fledglings

        self.eggs_laid = 0
        self.eggs_hatched = 0
        self.eggs_failed = 0
        self.chicks_dead = 0
        self.chicks_fledged = 0

    def condition_reset(self):

        self.t = 0
        self.eco_graph = nx.Graph()
        self.data_out = pd.DataFrame()

        self.eggless_pairs = []
        self.lam_lay = 1 
        self.lam_hatch = 0
        self.lam_death = 0
        self.lam_fledge = 0
        
        self.eggs = []
        self.hatched = []
        self.failed = []
        self.fledged = []
        self.deceased = []

        self.eggs_laid = 0
        self.eggs_hatched = 0
        self.eggs_failed = 0
        self.chicks_dead = 0
        self.chicks_fledged = 0
        
    
    def graph_setup(self):
        parent_node = 'Parent'
        child_nodes = ['Chick_1', 'Chick_2', 'Chick_3', 'Chick_4']

        for pair in range(1, self.osprey_pairs+1):
            for node in child_nodes:
                self.eco_graph.add_edge(f'{parent_node}_{pair}', f'{node}_{pair}')
                self.eco_graph.nodes[f'{node}_{pair}']['status'] = 'empty'
                #print(self.eco_graph.nodes[f'{node}_{pair}'])
        #print(self.eco_graph)

        for node in self.eco_graph.nodes:
            # G.neighbors(node) returns an iterator of connected nodes
            if re.match('Parent.*', node):
                connections = list(self.eco_graph.neighbors(node))
                #print(f'Node: {node} -> Connections: {connections}')

    def lay_rate_set(self):
        #Determine the remanining breeding pairs that have not laid eggs yet
        self.eggless_pairs = []
        self.lam_lay = 0
        for node in self.eco_graph.nodes:
            if re.match('Parent.*', node):
                child_1 = list(self.eco_graph.neighbors(node))[0]
                if self.eco_graph.nodes[child_1]['status'] == 'empty':
                    self.eggless_pairs.append(node)

        # On average, 3-20 days after courtship / nesting begins eggs will be laid
        self.lam_lay = len(self.eggless_pairs) * 0.1


    def hatch_rate_set(self):
        self.eggs = []
        self.lam_hatch = 0
        for node in self.eco_graph.nodes:
            if re.match('Chick.*', node):
                if self.eco_graph.nodes[node]['status'] == 'egg':
                    self.eggs.append(node)

        # ~ 35 days to hatch. 10/20 % of eggs fail to hatch, but are typically incubated for the same
        # amount of time. for this reason, will pull from a binomial distribution on each hatch event to determine
        # successful hatch vs dead chick
        self.lam_hatch = (1/35) * len(self.eggs) 

    def fledge_death_rate_set(self):
        # https://www.robertgroosphotography.com/home/fratricide-in-the-osprey-nest look to incorporate birth order?
        # average fledge time of ~ 50 days
        self.lam_fledge = (1/50) * len(self.hatched)
        self.lam_death = self.lam_fledge / (self.survival_prop / (1-self.survival_prop))

    def ocean_health(self):
        pass 

    def waiting_time(self, run: int):
        if self.lam_lay != 0:
            self.lay_rate_set()

        self.hatch_rate_set()
        self.fledge_death_rate_set()

        lam = sum([vars(self)[key] for key in vars(self).keys() if re.match('.*lam.*', key)])

        if lam == 0:
            return 'nothing', self.run_time - self.t

        rg = Generator(PCG64(self.master_seed+run))

        u1 = 1. - rg.random(1)
        tau = -np.log(u1) / lam

        probs = [vars(self)[key]/lam for key in vars(self).keys() if re.match('.*lam.*', key)]
        events  = [key for key in vars(self).keys() if re.match('.*lam.*', key)]
        selected_event = re.sub(r'lam\_', '', random.choices(events, weights=probs, k=1)[0])
        
        return selected_event, tau[0]

    def execute_step(self, run: int):

        event, tau = self.waiting_time(run)

        # egg laying event - select a random pair from our population of breeding pairs with no 
        # eggs. they will lay between 1 and 4 eggs
        if event == 'lay':
            pair = random.choice(self.eggless_pairs)
            number_of_eggs = random.choices([1,2,3,4], weights=[0.025, 0.025,0.8,0.05], k=1)[0]
            chicks = list(self.eco_graph.neighbors(pair))

            for i in range(1, 5):
                if i <= number_of_eggs:
                    self.eco_graph.nodes[chicks[i-1]]['status'] = 'egg'
                    self.eggs_laid += 1
                else:
                    self.eco_graph.nodes[chicks[i-1]]['status'] = 'not-taken'

        elif event == 'hatch':
            # https://pmc.ncbi.nlm.nih.gov/articles/PMC3497133/
            chick = random.choice(self.eggs)
            sucessful_hatch = random.choices(['y', 'n'], [0.9, 0.1], k=1)[0]

            if sucessful_hatch=='y':
                self.eco_graph.nodes[chick]['status'] = 'chick'
                self.hatched.append(chick)
                self.eggs_hatched += 1
            elif sucessful_hatch == 'n':
                self.eco_graph.nodes[chick]['status'] = 'failed hatch'
                self.failed.append(chick)
                self.eggs_failed += 1

        # Chicks dying or suriving to fledge
        elif event == 'death':
            chick = random.choice(self.hatched)
            self.hatched.remove(chick)
            self.eco_graph.nodes[chick]['status'] = 'dead'
            self.deceased.append(chick)
            self.chicks_dead += 1

        elif event == 'fledge':
            chick = random.choice(self.hatched)
            self.hatched.remove(chick)
            self.eco_graph.nodes[chick]['status'] = 'fledged'
            self.eco_graph.nodes[chick]['fledge_t'] = self.t
            self.fledged.append(chick)

            self.chicks_fledged += 1


        snapshot = pd.DataFrame([{'time': self.t, 
                                  'eggs': len(self.eggs),
                                  'failed': len(self.failed),
                                  'hatched': len(self.hatched),
                                  'died': len(self.deceased),
                                  'fledged': len(self.fledged)}])

        self.data_out = pd.concat([self.data_out, snapshot]).reset_index(drop=True)
        self.t = self.t + tau

    def runner(self, run: int):
        self.graph_setup()

        while self.t < 200:
            self.execute_step(run)

        #print(self.data_out)

        run_results = pd.DataFrame([{'run': run, 'total_eggs': self.eggs_laid, 'total_hatches': self.eggs_hatched, 'total_failed_eggs': self.eggs_failed, 'total_chicks_dead': self.chicks_dead, 'total_chicks_fledged': self.chicks_fledged}])
        return run_results
    
    def multi_run(self):    
        multi_run_results = pd.DataFrame()
        for i in range(0, self.nruns):
            print(i)
            run_results = self.runner(i)
            multi_run_results = pd.concat([multi_run_results, run_results]).reset_index(drop=True)
            self.condition_reset()

        print(multi_run_results)
        print(multi_run_results[['total_eggs', 'total_chicks_fledged', 'total_failed_eggs', 'total_chicks_dead']].mean())


def main():
    sim = eco_sim_basic(20, 200, 100, 1234)
    sim.multi_run()

if __name__ == '__main__':
    main()