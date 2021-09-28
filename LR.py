import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import math as math

daynight = {
    "D" : 1,
    "N" : 0
}
groups = ['Low', 'Med', 'High', 'Exp']
groups_value = {
    "Low": 3,
    "Med": 5,
    "High": 7,
    "Exp":10
}
satellite = {
    "Terra": 5,
    "Aqua": 10
}

class BasisFunction:
    def transform(self, data):
        # 2n Degree basis
        data['bright_track'] = data['brightness']*data['track']
        data['bright_scan'] = data['brightness']*data['scan']
        data['track_scan'] = data['track']*data['scan']
        data['bright_bright_t31'] = data['brightness']*data['bright_t31']
        data['daynight_brightness'] = data['daynight'] * data['brightness']
        data['daynight_track'] = data['daynight'] * data['track']
        data['daynight_scan'] = data['daynight'] * data['scan']
        data['daynight_confidence'] = data['daynight'] * data['confidence']
        data['brightness_2'] = data["brightness"].apply(lambda x: x**2)
        data['confidence'] = data["confidence"].apply(lambda x: x**2)
        data['bright_t31'] = data["bright_t31"].apply(lambda x: x**.7)
        data['scan_2'] = data["scan"].apply(lambda x: x**.7)
        data['track_2'] = data["track"].apply(lambda x: x**.7)
        data['daynight'] = data["daynight"].apply(lambda x: x**2)
        data['confidence_level_brightness'] = data['confidence_level'] * data['brightness']
        data['confidence_level_daynight'] = data['confidence_level'] * data['daynight']
        data['confidence_level_brightt31'] = data['confidence_level'] * data['bright_t31']
        data['confidence_level_scan'] = data['confidence_level'] * data['scan']
        data['acq_date_brightness'] = data['acq_date'] * data['brightness']
        data['acq_date_daynight'] = data['acq_date'] * data['daynight']


        # Higher degree baisis of some feature
        data['track_2*confidence_2'] = (data['track']**2)*(data['confidence']**2)
        data['confidence_2*bright_t31_2']=(data['confidence']**2)*(data['bright_t31']**2)
        data['track_3*confidence_2'] = (data['track']**3)*(data['confidence']**2)
        data['track_3*brightness'] = (data['track']**3)*(data['brightness']**2)
        data['track_3*brightness_daynight'] = (data['track']**3)*(data['brightness']**2) * data['daynight']
        data['track_3*bright_t31_2_daynight'] = (data['track']**3)*(data['bright_t31']**2) * data['daynight']
        data['track_3*confidence_2_daynight'] = (data['track']**3)*(data['confidence']**2) * data['daynight']
        data['track_3*confidence_2_brightness'] = (data['track']**3)*(data['confidence']**2) * data['brightness']
        data['track_3*confidence_2_2'] = (data['track_2*confidence_2'])**2
        data['track_3*brightness_daynight_2'] = data['track_3*brightness_daynight']**2
        return data 


class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self):
        self.min = {}
        self.max = {}
    def __call__(self, data, is_train=False):
    #     normalization
        if(is_train):
            self.min = data.min()
            self.max = data.max()
        return (data-self.min)/(self.max-self.min)

def get_features(csv_path, is_train=False, scaler=None):
    data = pd.read_csv(csv_path, usecols=['latitude', 'longitude', 'brightness', 'scan', 
                                          'track', 'acq_time', 'confidence', 'bright_t31', 'acq_date', 'daynight', 'satellite'])
    # Feature Engineering
    data['daynight'] = data['daynight'].replace(daynight)
    data['satellite'] = data['satellite'] .replace(satellite)
    data['acq_date'] = pd.to_datetime(data['acq_date']) - pd.to_datetime(data['acq_date'].min())
    data['acq_date'] = data['acq_date'].dt.days
    data['confidence_level'] = pd.qcut(data['confidence'], q=4, labels=groups)
    data['confidence_level'] = data['confidence_level'] .replace(groups_value)

    #     #scaling value of mostly correlated field
    basis = BasisFunction()
    data = basis.transform(data)
    
    if scaler:
        data = scaler.__call__(data, is_train)
        
    return data.to_numpy()

def get_targets(csv_path):
    data = pd.read_csv(csv_path)['frp']
    return data.to_numpy()


def analytical_solution(feature_matrix, targets, C=0.0):
    transpose = np.transpose(feature_matrix)
    tempmatrix = np.matmul(transpose,feature_matrix)-C*np.ones(len(transpose))
    tempinverse = np.linalg.inv(tempmatrix)
    temp2 = np.matmul(transpose,targets)
    return np.matmul(tempinverse, temp2)


def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error
    predictions = get_predictions(feature_matrix, weights)
    loss = mse_loss(feature_matrix, weights, targets)
    return loss

def get_predictions(feature_matrix, weights):
    return feature_matrix@weights

def mse_loss(feature_matrix, weights, targets):
    return np.square(np.subtract(targets, get_predictions(feature_matrix,weights))).mean()


def l2_regularizer(weights):
    return np.dot(weights, weights)


def loss_fn(feature_matrix, weights, targets, C=0.0):
    return mse_loss(feature_matrix, weights, targets) + C * l2_regularizer(weights);


def compute_gradients(feature_matrix, weights, targets, C=0.0):
    predicted = get_predictions(feature_matrix, weights)
    array = initialize_weights(len(weights))
    for k in range(len(weights)):
        result = 0.0;
        for i in range(len(feature_matrix)):
            result = result + np.dot((predicted[i] - targets[i]),feature_matrix[i][k])
        result = result + C * weights[k]
        result=result/len(feature_matrix)
        array[k]=result
    return array
                   
    
def sample_random_batch(feature_matrix, targets, batch_size):
    indices = np.random.choice(feature_matrix.shape[0], size = batch_size)
    return (feature_matrix[indices, :], targets[indices])


def initialize_weights(n):
    return np.zeros(n);


def update_weights(weights, gradients, lr):
    weights = np.subtract(weights , np.dot(lr,gradients))
    return weights

def early_stopping(min_not_change_count):
    # allowed to modify argument list as per your need
    # return True or False
    return  min_not_change_count >= 500


def plot_trainsize_losses(x, y, xlabel, ylabel, figname, filename="test.jpg"):
    '''
    Description:
    plot losses on the development set instances as a function of training set size
    '''

    '''
    Arguments:
    # you are allowed to change the argument list any way you like 
    '''
    fig = plt.figure()
    plt.plot(x, y, color="blue")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.suptitle(figname, fontsize=20)
    plt.savefig(filename)
    
    
def do_gradient_descent(train_feature_matrix,
                        train_targets,
                        dev_feature_matrix,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=50000,
                        eval_steps=5):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights

    a sample code is as follows --
    '''
    min_loss = np.inf
    min_loss_unchange_count = 0
    best_weight = initialize_weights(train_feature_matrix.shape[1])
    weights = initialize_weights(train_feature_matrix.shape[1])
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)
    print("step {} \t dev loss: {} \t train loss: {}".format(0, dev_loss, train_loss))
    for step in range(1, max_steps + 1):

        # sample a batch of features and gradients
        features, targets = sample_random_batch(train_feature_matrix, train_targets, batch_size)
        
        # compute gradients
        gradients = compute_gradients(features, weights, targets, C)

        # update weights
        weights = update_weights(weights, gradients, lr)

        if step % eval_steps == 0:
            dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            print("step {} \t dev loss: {} \t train loss: {}".format(step, dev_loss, train_loss))

            
            '''
            implement early stopping etc. to improve performance.
            '''
            if min_loss>dev_loss:
                min_loss = dev_loss
                min_loss_unchange_count = 0
                best_weight = weights
            else:
                min_loss_unchange_count +=1
                if early_stopping(min_loss_unchange_count):
                    break
            
        
    return best_weight


if __name__ == '__main__':

    scaler = Scaler()
    train_features, train_targets = get_features('train.csv', True, scaler), get_targets('train.csv')
    dev_features, dev_targets = get_features('dev.csv', False, scaler), get_targets('dev.csv')
    a_solution = analytical_solution(train_features, train_targets, C=1e-34)
    print('evaluating analytical_solution...')
    dev_loss = do_evaluation(dev_features, dev_targets, a_solution)
    train_loss = do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
    test = get_features('test.csv', True, scaler)
    df = pd.DataFrame(get_predictions(test, a_solution), columns=['frp'])
    df['ID'] = np.arange(0, len(df), 1)
    df.to_csv("output.csv", columns=['ID', 'frp'], index=None)


    
    
    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features,
                                                train_targets,
                                                dev_features,
                                                dev_targets,
                                                lr=0.25,
                                                C=0,
                                                batch_size=32,
                                                max_steps=180000,
                                                eval_steps=1000)

    print('evaluating iterative_solution...')
    dev_loss_gd= do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss = do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss_gd))



    x=[]
    y=[]
    for i in range(5000, train_features.shape[0], 5000):
        a_solution = analytical_solution(train_features[:i], train_targets[:i], C=1e-8)
        y.append(do_evaluation(dev_features, dev_targets, a_solution))
        x.append(i)
    x.append(train_features.shape[0])
    y.append(dev_loss)
    plot_trainsize_losses(x,y,"training data size", "MSE", 
                            "MSE with respect to training size","analytical.jpg")

    C_values = []
    loss_values = []
    lamda_values = [ 0, 1e-8, 1e-6, 1e-5, 0.075, 0.1 , 0.25, 0.5, .75, 1]
    for c in lamda_values:
        gradient_descent_soln = do_gradient_descent(train_features, 
                            train_targets, 
                            dev_features,
                            dev_targets,
                            lr=0.25,
                            C=c,
                            batch_size=32,
                            max_steps=50000,
                            eval_steps=5)
        dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
        C_values.append(c)
        loss_values.append(dev_loss)

    plot_trainsize_losses(C_values, loss_values, "(C) value", "Dev Loss","Lambda vs Dev Loss","losswithC")
