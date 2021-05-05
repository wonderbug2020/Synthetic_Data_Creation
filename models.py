#Create the train/test split
def get_split(X,y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
    return(X_train, X_test, y_train, y_test)

#Standardize the data using StandardScaler
def standardize(X_train,X_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return (X_train,X_test)

#Baseline Random forest model
def rfc(X_train,y_train,X_test):
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state = 0)
    rfc.fit(X_train,y_train)
    y_pred = rfc.predict(X_test)
    return y_pred

#Baseline Logistic regression model
def def log_reg(X_train,y_train,X_test):
    from sklearn.linear_model import LogisticRegression
    log_cla = LogisticRegression(random_state = 0)
    log_cla.fit(X_train, y_train)
    y_pred = log_cla.predict(X_test)
    return y_pred

def get_results(y_test, y_pred):
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, log_pred)
    accuracy_score(y_test, log_pred)
    return (cm, accuracy_score)

def run_rfc(X,y):
    X_train, X_test, y_train, y_test = get_split(X,y)
    y_pred = rfc(X_train,y_train,X_test)
    cm, accuracy_score = get_results(y_test, y_pred)
    print(cm)
    print(accuracy_score)

def run_log_reg(X,y):
    X_train, X_test, y_train, y_test = get_split(X,y)
    y_pred = leg_reg(X_train,y_train,X_test)
    cm, accuracy_score = get_results(y_test, y_pred)
    print(cm)
    print(accuracy_score)
