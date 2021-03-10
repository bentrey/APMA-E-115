import numpy as np
import matplotlib.pyplot as plt

class agent:
    
    def __init__ (self, x, y, mobility):
        self.x = x
        self.y = y
        self.mobility = mobility*np.random.random()
        self.contacts = []
        self.status = 's'
    
    def infect(self):
        if self.status == 's':
            self.status = 'i'
    
    def recover(self):
        if self.status == 'i':
            self.status = 'r'
    
def logistic(x):
    return np.exp(x)/(1+np.exp(x))
    
def distance(r):
    return np.sqrt((self.x-r[0])**2+(self.y-r[1])**2)

def census(population):
    
    #create an array of statuses
    statuses = np.array([individual.status for individual in population])

    #vectorized for loops counting statuses 
    s = np.sum(statuses == 's')
    i = np.sum(statuses == 'i')
    r = np.sum(statuses == 'r')
    
    return np.array([s, i, r])


def epidemic1D(pt, gamma, population, dt):

    N = population.size
    
    for n in range(N):
        
        #check if the indidual is infected
        if population[n].status=='i':
            
            #infect the susceptible others with a certain probability
            for index in population[n].contacts:
                indicies = np.random.rand()
                if np.random.rand()<pt*dt:
                    population[index].infect()
                    
            #put the infected individual in the recovered state
            if np.random.random()<gamma*dt:
                population[n].recover()
            
    return population
            

def simulate1D(pt, mobility, gamma, N, radius, total_population, dt):
    
    #list of the sir populations by week
    sir = []
    
    #population_centers = np.array([ [20*np.random.random(),20*np.random.random()] for n in range(N) ])
    population_centers = np.array([[ 6.42466026, 14.02406142], [ 8.72068026,  3.62701442], [16.28406737,  9.07873334],\
       [18.53413048, 15.55366608], [14.01541958, 14.0999397 ]])
    indices = np.random.choice(N,size=total_population)
    individual_locations = population_centers[indices]+np.random.normal(0,radius,(total_population,2))
    xs = individual_locations[:,0]
    ys = individual_locations[:,1]
    
    #creates an array of the class agent
    population = np.array([agent(xs[n], ys[n], mobility) for n in range(total_population)])
    
    #create network
    for n in range(total_population):
        for m in range(total_population):
            d = np.sqrt((population[n].x-population[m].x)**2+(population[n].y-population[m].y)**2)
            log = np.exp(-d/population[n].mobility)/(1+np.exp(-d/population[n].mobility))
            if np.random.random() < log and d > 0.00001:
                population[n].contacts.append(m)
                population[m].contacts.append(n)
    
    #randomly infect 1 percent
    indices = np.random.choice(np.arange(total_population),total_population//100,replace=False)
    for index in indices:
        population[index].infect()
        
    #record initial populations
    state = census(population)
    sir.append(state)
    
    #simulate the remaining weeks
    while len(sir)<int(100/dt):
        new_population=epidemic1D(pt, gamma, population, dt)
        state = census(new_population)
        sir.append(state)
        population = new_population
        
    return np.array(sir), population, individual_locations