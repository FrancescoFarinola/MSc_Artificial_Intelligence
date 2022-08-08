import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neuralnet as nn

figsize=(10,5)
def load_data(data_folder):
    # Read the CSV files
    fnames = ['train_FD001', 'train_FD002', 'train_FD003', 'train_FD004']
    cols = ['machine', 'cycle', 'p1', 'p2', 'p3'] + [f's{i}' for i in range(1, 22)]
    datalist = []
    nmcn = 0
    for fstem in fnames:
        # Read data
        data = pd.read_csv(f'{data_folder}/{fstem}.txt', sep=' ', header=None)
        # Drop the last two columns (parsing errors)
        data.drop(columns=[26, 27], inplace=True)
        # Replace column names
        data.columns = cols
        # Add the data source
        data['src'] = fstem
        # Shift the machine numbers
        data['machine'] += nmcn
        nmcn += len(data['machine'].unique())
        # Generate RUL data
        cnts = data.groupby('machine')[['cycle']].count()
        cnts.columns = ['ftime']
        data = data.join(cnts, on='machine')
        data['rul'] = data['ftime'] - data['cycle']
        data.drop(columns=['ftime'], inplace=True)
        # Store in the list
        datalist.append(data)
    # Concatenate
    data = pd.concat(datalist)
    # Put the 'src' field at the beginning and keep 'rul' at the end
    data = data[['src'] + cols + ['rul']]
    return data

def split_by_field(data, field):
    res = {}
    for fval, gdata in data.groupby(field):
        res[fval] = gdata
    return res

def partition_by_machine(data, tr_machines):
    # Separate
    tr_machines = set(tr_machines)
    tr_list, ts_list = [], []
    for mcn, gdata in data.groupby('machine'):
        if mcn in tr_machines:
            tr_list.append(gdata)
        else:
            ts_list.append(gdata)
    # Collate again
    tr_data = pd.concat(tr_list)
    ts_data = pd.concat(ts_list)
    return tr_data, ts_data
def plot_training_history(history, 
        figsize=(16,8), autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    for i in history.keys():
        plt.plot(history[i]['train_acc'], label=i)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rul(pred=None, target=None,
        stddev=None,
        q1_3=None,
        same_scale=True,
        figsize=figsize, title=None):
    plt.figure(figsize=figsize)
    if target is not None:
        plt.plot(range(len(target)), target, label='target',
                color='tab:orange')
    if pred is not None:
        if same_scale or target is None:
            ax = plt.gca()
        else:
            ax = plt.gca().twinx()
        ax.plot(range(len(pred)), pred, label='pred',
                color='tab:blue')
        if stddev is not None:
            ax.fill_between(range(len(pred)),
                    pred-stddev, pred+stddev,
                    alpha=0.3, color='tab:blue', label='+/- std')
        if q1_3 is not None:
            ax.fill_between(range(len(pred)),
                    q1_3[0], q1_3[1],
                    alpha=0.3, color='tab:blue', label='1st/3rd quartile')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

data_folder = 'data'
data = load_data(data_folder)

dt_in = list(data.columns[3:-1])
data_sv = data.copy()
data_sv[dt_in] = (data_sv[dt_in] - data_sv[dt_in].mean()) / data_sv[dt_in].std()
data_sv_dict = split_by_field(data_sv, field='src')
print('{{{}}}'.format(', '.join(f'{k}: ...' for k in data_sv_dict.keys())))

dt = data_sv_dict['train_FD001']    

#separate train and test set
tr_ratio = 0.8
np.random.seed(42)
machines = dt.machine.unique()
np.random.shuffle(machines)
sep = int(tr_ratio * len(machines))
tr_mcn = machines[:sep]
ts_mcn = machines[sep:]
tr, ts = partition_by_machine(dt, tr_mcn)

#separate train and validation set
tr_machines = tr.machine.unique()
tv_ratio = 0.8
sep1 = int(tv_ratio * len(tr_machines))
tr_mcn = machines[:sep1]
tv_mcn = machines[sep1:]
tr, tv = partition_by_machine(tr, tr_mcn)

#standardize values
trmean = tr[dt_in].mean()
trstd = tr[dt_in].std().replace(to_replace=0, value=1) # handle static fields
ts_s = ts.copy()
ts_s[dt_in] = (ts_s[dt_in] - trmean) / trstd
tr_s = tr.copy()
tr_s[dt_in] = (tr_s[dt_in] - trmean) / trstd
tv_s = tv.copy()
tv_s[dt_in] = (tv_s[dt_in] - trmean) / trstd

#normalize rul
trmaxrul = tr['rul'].max()
ts_s['rul'] = ts['rul'] / trmaxrul 
tr_s['rul'] = tr['rul'] / trmaxrul
tv_s['rul'] = tv['rul'] / trmaxrul

x_train = tr_s[dt_in].to_numpy()
y_train = tr_s['rul'].to_numpy()
x_test = ts_s[dt_in].to_numpy()
y_test = ts_s['rul'].to_numpy()
x_val = tv_s[dt_in].to_numpy()
y_val = tv_s['rul'].to_numpy()


def best_opt():
    
    optimizer = ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']
    histories = dict()
    model = []
    for i, opt in enumerate(optimizer):
        print('Training with ' + opt + ' optimizer:')
        model.append(nn.NeuralNetwork())
        model[i].add(nn.Layer(32, activation='relu'))
        model[i].add(nn.Layer(32, activation='relu'))
        model[i].add(nn.Layer(1, activation='linear'))
        stop = nn.EarlyStopping(patience=5, delta=1e-3, restore_weights=True)
        model[i].compile(epochs=30, learning_rate=1e-3, loss='mse', optimizer=opt, earlystop=stop)
        model[i].fit(x_train, y_train, x_val, y_val, batch_size=32)
        histories[opt] = model[i].history
    min_train_loss = 100
    for i, opt in enumerate(histories.keys()):
        if histories[opt]['train_acc'][-1] < min_train_loss:
            min_train_loss = histories[opt]['train_acc'][-1]
            best_model = model[i]
    print('Best optimizer: ' + best_model.optimizer)
    return histories, best_model
        
history, model = best_opt()

plot_training_history(history,figsize=figsize)

ts_pred = model.forward(x_test).ravel() * trmaxrul
y_test = y_test * trmaxrul
stop = 3000
plot_rul(ts_pred[:stop], y_test[:stop], figsize=figsize, title='Test set predictions with '+ model.optimizer+' optimizer')
