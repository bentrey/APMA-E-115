import urllib.request
import pandas as pd
import datetime

def census(sir):
    s = np.sum(sir[:,0])
    i = np.sum(sir[:,1])
    r = np.sum(sir[:,2])
    return np.array([s, i, r])

def loss_function(sir, df, state_data, weights, date):
    loss = 0
    state_list = df['State or territory'].to_numpy()
    sir_data = np.zeros_like(sir)
    grads = {'alpha':np.zeros((51,51)), 'beta':np.zeros((51,51)), 'gamma':np.zeros((51,51))}
    beta = weights['beta']
    gamma = weights['gamma']
    N = np.sum(sir)
    for n in range(len(state_list)):
        state = state_list[n]
        state_df = state_data[state]
        #get population in state
        population = int(df[df['State or territory'] == state].iloc[0,3])
        #get positive from data
        positive_data = state_df[state_df['date']==date].positive.to_numpy()[0]
        #get recovered from data
        recovered_data = state_df[state_df['date']==date].recovered.to_numpy()[0]+state_df[state_df['date']==date].death.to_numpy()[0]
        sir_data[n,0] = population - positive_data
        sir_data[n,1] = positive_data - recovered_data
        sir_data[n,2] = recovered_data
    loss += np.sum((beta/N*(sir_data[:,1]*sir_data[:,0]-sir[:,1]*sir[:,0]))**2)
    loss += np.sum((beta/N*(-sir_data[:,1]*sir_data[:,0]+sir[:,1]*sir[:,0])+\
             gamma*(sir_data[:,1]-sir[:,1]))**2)
    loss += np.sum((gamma*(sir_data[:,1]-sir[:,1]))**2)
    grads['beta'] = -2*(np.outer(sir[:,0],sir[:,1])-np.outer(sir_data[:,0],sir_data[:,1]))**2*beta/N**2
    grads['gamma'] = (4*gamma*(sir[:,1]-sir_data[:,1])**2/
        +2*(-sir[:,1]+sir_data[:,1])*\
        (sir[:,0]*np.matmul(beta,sir[:,1])/N-sir_data[:,0]*np.matmul(beta,sir_data[:,1])/N))
    return loss, grads

def update_populations(sir, weights, dt):
    beta = weights['beta']
    gamma = weights['gamma']
    N = np.sum(sir)
    return_sir = sir.copy()
    return_sir[:,0] += -sir[:,0]*np.matmul(beta,sir[:,1])*dt/N
    return_sir[:,1] += sir[:,0]*np.matmul(beta,sir[:,1])*dt/N - gamma*sir[:,1]*dt
    return_sir[:,2] += gamma*sir[:,1]*dt
    return return_sir

def run_simulation(dt, trials=10, learning_rate=0.001):
    
    start_date = '2020-05-01'
    end_date = '2020-06-01'

    start_time = datetime.datetime.strptime(start_date,'%Y-%m-%d')
    end_time = datetime.datetime.strptime(end_date,'%Y-%m-%d')
    cycles = int((end_time-start_time).total_seconds()/dt)
    
    url = 'https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States_by_population'
    file = urllib.request.urlopen(url).read()
    tables = str(file).split('<table')
    for table in tables:
        if '39,368,078' in table:
            pop_table = '<table'+table.split('table>')[0]+'table>'
            pop_table = pop_table.replace('\\n','')
            df = pd.read_html(pop_table)[0]
            df.columns = df.columns.get_level_values(1)
    
    df = df[df['State or territory'] !='Puerto Rico']
    df = df[df['State or territory'] !='Guam']
    df = df[df['State or territory'] !='U.S. Virgin Islands']
    df = df[df['State or territory'] !='Northern Mariana Islands']
    df = df[df['State or territory'] !='American Samoa']
    df = df[df['State or territory'] !='Contiguous United States']
    df = df[df['State or territory'] !='The fifty states and D.C.']
    df = df[df['State or territory'] !='Total United States']
    df = df[df['State or territory'] !='The fifty states']
    
    states = df['State or territory'].to_numpy()
    
    weights = {'beta':np.random.random((len(states),len(states)))/10**7, 'gamma':np.random.random(len(states))/10**9}
    
    states_data = {}
    for state in states:
        states_data[state] = pd.read_csv('C:/Users/bentr/Documents/harvard/MathematicalModeling/Project_1/State_Data/'\
            +state.lower().replace(' ','-')+'-history.csv').fillna(0)
    
    loss = np.zeros(trials)
    
    for trial in range(trials):
        
        sir = np.zeros((df.shape[0],3),dtype=float)
        sir[:,0] = df['EstimatedJuly 1, 2020[6]'].to_numpy(dtype=float)
    
        for n in range(len(states)):
            state_data = states_data[states[n]]
            positive = state_data[state_data['date']==start_date].positive.to_numpy(dtype=float)[0]
            recovered = state_data[state_data['date']==start_date].recovered.to_numpy(dtype=float)[0]
            sir[n,0] += -positive  
            sir[n,1] += positive - recovered
            sir[n,2] += recovered
            
        for cycle in range(cycles):
            sir = update_populations(sir, weights, dt)
            
        trial_loss, grads = loss_function(sir, df, states_data, weights, end_date)
        
        weights['beta'] -= learning_rate*grads['beta']
        weights['gamma'] -= learning_rate*grads['gamma']
        loss[trial] = trial_loss
    
    start_date = '2020-05-01'
    end_date = '2020-07-01'

    start_time = datetime.datetime.strptime(start_date,'%Y-%m-%d')
    end_time = datetime.datetime.strptime(end_date,'%Y-%m-%d')
    cycles = int((end_time-start_time).total_seconds()/dt)
    
    sir[:,0] = df['EstimatedJuly 1, 2020[6]'].to_numpy(dtype=float)
    
    return_sir = np.zeros((cycles,3))
    
    for n in range(len(states)):
        state_data = states_data[states[n]]
        positive = state_data[state_data['date']==start_date].positive.to_numpy(dtype=float)[0]
        recovered = state_data[state_data['date']==start_date].recovered.to_numpy(dtype=float)[0]
        sir[n,0] += -positive  
        sir[n,1] += positive - recovered
        sir[n,2] += recovered
        
    return_sir[0,0] = np.sum(sir[:,0])
    return_sir[0,1] = np.sum(sir[:,1])
    return_sir[0,2] = np.sum(sir[:,2])
    
    for cycle in range(cycles):
        sir = update_populations(sir, weights, dt)
        return_sir[cycle] = census(sir)
        
    return return_sir, loss, weights
        