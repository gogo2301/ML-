import numpy as np
import util
import math

from linear_model import LinearModel
# def fit(self,x, y):
#         """Fit a GDA model to training set given by x and y.

#         Args:
#             x: Training example inputs. Shape (m, n).
#             y: Training example labels. Shape (m,).

#         Returns:
#             theta: GDA model parameters.
#         """
#         # *** START CODE HERE ***
#         def mean_1(x,y):
#             l=np.zeros(2)
#             e=0
#             a,b=np.shape(x)
#             for i in range(a):
#                 if y[i]==1:
#                     l+=x[i,:]
#                     e+=1
#             return l/e   
#         def mean_0(x,y):
#             l=np.zeros(2)
#             e=0
#             a,b=np.shape(x)
#             for i in range(a):
#                 if y[i]==0:
#                     l+=x[i,:]
#                     e+=1
#             return l/e
#         def phi(y):
#             a,b=np.shape(x)
#             e=0
#             for i in range(a):
#                 if y[i]==1:
#                     e+=1
#             return e/a
#         def covar(x,y):
#             cov=np.array([[0.0,0.0],[0.0,0.0]])
#             a,b=np.shape(x)
#             m1=np.array([mean_1(x,y)])
#             m0=np.array([mean_0(x,y)])
#             for i in range(a):
#                 if y[i]==1:
#                     m1=np.array([x[i,:]])-np.array([mean_1(x,y)])
#                     cov+=m1.T@m1
#                 else:
#                     m0=np.array([x[i,:]])-np.array([mean_0(x,y)])
#                     cov+=m0.T@m0
#             return cov/a
#         return mean_1(x,y)@np.linalg.inv(covar(x,y))+(mean_0(x,y).T@mean_0(x,y))/2-(mean_1(x,y).T@mean_1(x,y))/2-mean_0(x,y)@np.linalg.inv(covar(x,y))
#         # *** END CODE HERE ***
# def mean_1(x,y):
#     l=np.zeros(2)
#     e=0
#     a,b=np.shape(x)
#     for i in range(a):
#         if y[i]==1:
#             l+=x[i,:]
#             e+=1
#     return l/e   
# def mean_0(x,y):
#     l=np.zeros(2)
#     e=0
#     a,b=np.shape(x)
#     for i in range(a):
#          if y[i]==0:
#            l+=x[i,:]
#            e+=1
#     return l/e
# def covar(x,y):
#     cov=np.array([[0.0,0.0],[0.0,0.0]])
#     a,b=np.shape(x)
#     m1=np.array([mean_1(x,y)])
#     m0=np.array([mean_0(x,y)])
#     for i in range(a):
#         if y[i]==1:
#              m1=np.array([x[i,:]])-np.array([mean_1(x,y)])
#              cov+=m1.T@m1
#         else:
#             m0=np.array([x[i,:]])-np.array([mean_0(x,y)])
#             cov+=m0.T@m0
#     return cov/a
def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        def mean_1(x,y):
            l=np.zeros(2)
            e=0
            a,b=np.shape(x)
            for i in range(a):
                if y[i]==1:
                    l+=x[i,:]
                    e+=1
            return l/e   
        def mean_0(x,y):
            l=np.zeros(2)
            e=0
            a,b=np.shape(x)
            for i in range(a):
                if y[i]==0:
                    l+=x[i,:]
                    e+=1
            return l/e
        def phi(y):
            a,b=np.shape(x)
            e=0
            for i in range(a):
                if y[i]==1:
                    e+=1
            return e/a
        def covar(x,y):
            cov=np.array([[0.0,0.0],[0.0,0.0]])
            a,b=np.shape(x)
            m1=np.array([mean_1(x,y)])
            m0=np.array([mean_0(x,y)])
            for i in range(a):
                if y[i]==1:
                    m1=np.array([x[i,:]])-np.array([mean_1(x,y)])
                    cov+=m1.T@m1
                else:
                    m0=np.array([x[i,:]])-np.array([mean_0(x,y)])
                    cov+=m0.T@m0
            return cov/a
        theta_proto=np.linalg.inv(covar(x,y))@(np.array([mean_1(x,y)]).T)
        theta_proto-=np.linalg.inv(covar(x,y))@(np.array([mean_0(x,y)]).T)
        print(theta_proto)
        phivec=((np.array([mean_0(x,y)])@np.linalg.inv(covar(x,y)))@np.array([mean_0(x,y)]).T)/2
        phivec-=((np.array([mean_1(x,y)])@np.linalg.inv(covar(x,y)))@np.array([mean_1(x,y)]).T)/2
        phivec-=np.array([[math.log((1-phi(y))/phi(y))]])
        print(phivec)
        theta=np.append(phivec,theta_proto)
        print(theta)
        return theta
        # *** END CODE HERE ***

    def predict(self, x,y,theta):
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
        l=0
        for i in range(a):
            if empt[i]==y[i]:
                l+=1
        return empt,(l/a)*100
        # *** END CODE HERE
# x_train, y_train =util.load_dataset("C:\Desktop\ML\PS1 e\data e\data_e_train.csv",add_intercept=False)
# x_train1, y_train1=util.load_dataset("C:\Desktop\ML\PS1 e\data e\data_e_train.csv",add_intercept=True)
# x_pred,y_pred=util.load_dataset("C:\Desktop\ML\PS1 e\data e\data_e_predict.csv",add_intercept=True)
# a=GDA()
# theta=a.fit(x_train,y_train)
# # print(x_train1@(np.array([theta])).T)
# util.plot(x_train1,y_train1,theta,save_path="C:\Desktop\ML\PS1 e")
# print(a.predict(x_pred,y_pred,theta))