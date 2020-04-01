from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/home/diabetes', methods=['GET','POST'])
def diabetes():
    return render_template('diabetes.html')

@app.route('/home/diabetes/predict_diabetes', methods=['GET','POST'])
def predict_diabetes():
    features = [x for x in request.form.values()]
    print(features)
    username = features[0]
    print(username)
    input_list = features[1:]
    input_list = [int(x) for x in input_list]
    print(input_list)
    model = input_list[-1]
    input_list = input_list[:-1]
    return render_template('predict_diabetes.html', mylist=mylist(input_list,model))

def mylist(arr,a):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    arr = [np.array(arr)]
    print(arr)
    
    dataset =pd.read_csv('data/diabetes.csv')
    X=dataset.iloc[:,0:8].values
    y=dataset.iloc[:,8].values
    
    from sklearn.preprocessing import LabelEncoder
    y= LabelEncoder().fit_transform(y)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=0)
    
    from sklearn.preprocessing import StandardScaler
    sc_X= StandardScaler()
    X_train=sc_X.fit_transform(X_train)
    X_test=sc_X.transform(X_test)
    
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    
    Y_pred = classifier.predict(X_test)
    print(X_test)
    
    from  sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,Y_pred)
    
    pred = classifier.predict(arr)[0]
    print(type(pred))
    print(pred)
    if pred == 1:
        ans = "You will get it dude"
    else:
        ans = "Chill dude"    

    a = round((cm[0][0]+cm[1][1])*100/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]),2)
    b = round((cm[0][0])*100/(cm[0][0]+cm[1][0]),2)
    v = round((cm[0][0])*100/(cm[0][0]+cm[0][1]),2)
    d = round(2*(v*b)/(v+b),2)
    result = [a,b,v,d]
    print(result)
    result.append(ans)

    from sklearn.decomposition import PCA
    pca= PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance= pca.explained_variance_ratio_

    print(explained_variance)

    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)

    Y_pred = classifier.predict(X_test)

    from matplotlib.colors import ListedColormap
    X_set,y_set=X_test,y_test
    X1,X2= np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
    plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap =ListedColormap(('red','green')))
    plt.xlim(X1.min(),X1.max())
    plt.ylim(X2.min(),X2.max())
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
    plt.title('PCA for Selected Model')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # plt.legend()
    plt.savefig('static/images/model.png')

    return result   

@app.route('/home/diabetes/predict_diabetes/response', methods=['GET','POST'])
def diabetes_response():
    res = [x for x in request.form.values()]
    print(res)

    return render_template('diabetes.html')

if __name__ == "__main__":
    app.run(debug=True)