import numpy as np
import util

import p01e_gda
import p01


x1,y1=util.load_dataset("C:\Desktop\ML\PS1 f\data1\data_e_train.csv",add_intercept=True)
x2,y2=util.load_dataset("C:\Desktop\ML\PS1 f\data1\data_e_train.csv",add_intercept=False)
x3,y3=util.load_dataset("C:\Desktop\ML\PS1 f\data1\data_e_train.csv",add_intercept=True)

a1=p01e_gda.GDA()
theta1=a1.fit(x2,y2)
a2=p01.LogisticRegression()
theta2=a2.fit(x1,y1)
util.plot(x3,y3,theta2,theta1,save_path="C:\Desktop\ML\PS1 f")






