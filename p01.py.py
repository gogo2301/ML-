import numpy as np
import util
from linear_model import LinearModel

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.
    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset('C:\Desktop\ML\data/data.csv', add_intercept=True)

    # *** START CODE HERE ***
    # a=LogisticRegression()
    # theta=a.fit(x_train,y_train)
    # util.plot(x_train,y_train,theta,'c:\Desktop\ML')
    
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.
    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        def sigmoid(x,theta):
            return 1/(1+np.exp(-(np.dot(x,theta))))
        def hessian(x,theta):
            d11=np.sum(sigmoid(x,theta)*(1-sigmoid(x,theta))*x[:,0]*x[:,0])
            d12=np.sum(sigmoid(x,theta)*(1-sigmoid(x,theta))*x[:,0]*x[:,1])
            d13=np.sum(sigmoid(x,theta)*(1-sigmoid(x,theta))*x[:,0]*x[:,2])
            d21=np.sum(sigmoid(x,theta)*(1-sigmoid(x,theta))*x[:,1]*x[:,0])
            d22=np.sum(sigmoid(x,theta)*(1-sigmoid(x,theta))*x[:,1]*x[:,1])
            d23=np.sum(sigmoid(x,theta)*(1-sigmoid(x,theta))*x[:,1]*x[:,2])
            d31=np.sum(sigmoid(x,theta)*(1-sigmoid(x,theta))*x[:,2]*x[:,0])
            d32=np.sum(sigmoid(x,theta)*(1-sigmoid(x,theta))*x[:,2]*x[:,1])
            d33=np.sum(sigmoid(x,theta)*(1-sigmoid(x,theta))*x[:,2]*x[:,2])
            return [[d11,d12,d13],[d21,d22,d23],[d31,d32,d33]]
        def gradient(x,y,theta):
            d11=np.sum(y*x[:,0]-x[:,0]*(sigmoid(x,theta)))
            d12=np.sum(y*x[:,1]-x[:,1]*(sigmoid(x,theta)))
            d13=np.sum(y*x[:,2]-x[:,2]*(sigmoid(x,theta)))
            return np.array([[d11],[d12],[d13]])
        j=0
        theta_prev=np.zeros(3)
        delta=100
        while delta>self.eps and self.max_iter>j:
            theta_new=theta_prev+((np.linalg.inv(hessian(x,theta_prev))@gradient(x,y,theta_prev)).T)[0]
            delta=np.max(abs(theta_new-theta_prev))
            theta_prev=theta_new
            j+=1
        return theta_prev
        # *** END CODE HERE ***

    def predict(self, x,theta,y):
        """Make a prediction given new inputs x.
        Args:
            x: Inputs of shape (m, n).
        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        empt=np.array([])
        a,b=np.shape(x)
        for i in range(a):
            if np.sum(x[i,:]*theta)>0:
                empt=np.append(empt,[1])
            else:
                empt=np.append(empt,[0])
        mean=0
        for i in range(len(y)):
            if empt[i]==y[i]:
                mean+=1
        accuracy=(mean/len(y))*100
        return empt,accuracy
    
        # *** END CODE HERE ***
x_train, y_train = util.load_dataset("C:\Desktop\ML\PS1 e\data e\data_e_train.csv", add_intercept=True)
x_pred, y_pred = util.load_dataset("C:\Desktop\ML\PS1 e\data e\data_e_predict.csv",add_intercept=True)

    # *** START CODE HERE ***
a=LogisticRegression()
theta=a.fit(x_train,y_train)
print(theta)
# print(x_train)
util.plot(x_train,y_train,theta,save_path='C:\Desktop\ML\PS1 b')
pred=a.predict(x_pred,theta,y_pred)
print(pred)

# 3.3511764604087624,47.752517184092525,0.0
