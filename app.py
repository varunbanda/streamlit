import streamlit as st
# import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.svm import SVC
from alibi.explainers import AnchorTabular
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from alibi.confidence import TrustScore
st.set_option('deprecation.showPyplotGlobalUse', False) #Used to disable warnings

st.title('Blackbox Models Explained')
dataset_choice = st.sidebar.selectbox('Select a dataset:', ('Iris','Wine', 'Cancer'))
if dataset_choice == 'Wine':
    selected_dataset = load_wine()
    selected_dataset_df = load_wine(as_frame=True)
elif dataset_choice == 'Cancer':
    selected_dataset = load_breast_cancer()
    selected_dataset_df = load_breast_cancer(as_frame=True)
else:
    selected_dataset = load_iris()
    selected_dataset_df = load_iris(as_frame=True)
dataset = selected_dataset.data
feature_names = selected_dataset.feature_names
class_names = list(selected_dataset.target_names)
df_data = selected_dataset_df.data.copy() # Dataset for displaying
df_data['target'] = selected_dataset_df.target
# df_data['target'].replace({0:'malignant', 1:'benign'}, inplace=True)
st.write("""
### Dataset:""")
st.dataframe(df_data)
X_train, X_test, y_train, y_test = train_test_split(selected_dataset.data, selected_dataset.target, random_state = 42)
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(selected_dataset_df.data, df_data['target'], random_state = 42)
st.write("""
### Dimensions of train and test sets:""")
st.write('X_train shape:', X_train.shape)
st.write('X_test shape:', X_test.shape)
st.write('y_train shape:', y_train.shape)
st.write('y_test shape:', y_test.shape)
st.write("""### Target imbalance in train:""")
plt.hist(y_train)
st.pyplot()
# st.sidebar.write("Select a model:")
# print(' Till here ')
selected_model = st.sidebar.selectbox('Select a model:', ('BB1', 'BB2', 'BB3'))
# st.sidebar.write(selected_model)
# st.sidebar.write("Select an instance:")
X_test_df['target'] = y_test_df
if selected_model == "BB1":
    clf = SVC(probability=True, random_state=42)
elif selected_model == 'BB2':
    clf = DecisionTreeClassifier(random_state=42)
else:
    clf = RandomForestClassifier(random_state=42)
# st.sidebar.write(selected_model)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
st.write("""### Metrics:""")
st.write('Train accuracy:', accuracy_score(y_train,clf.predict(X_train)))
st.write('Test accuracy:', accuracy_score(y_test, clf.predict(X_test)))
confusion_matrix(y_test, y_pred)
st.write('Confusion matrix:')
plot_confusion_matrix(clf, X_test, y_test)
st.pyplot()
# st.write(classification_report(y_test, y_pred))
predict_fn = lambda x: clf.predict_proba(x)
explainer = AnchorTabular(predict_fn, feature_names)
explainer.fit(X_train)
idx = st.sidebar.slider(label='Select an instance:',min_value=1,max_value=len(y_test))
st.write("""### Selected instance:""")
st.write(X_test_df.iloc[[idx-1]], height=150)
print(y_train_df.iloc[[idx-1]])
st.write('Prediction: ', class_names[explainer.predictor(X_test[idx-1].reshape(1, -1))[0]])
st.write("""### Prediction Explained:""")
with st.spinner('Calculating'):
    explanation = explainer.explain(X_test[idx-1], threshold=0.70)
    st.write('Anchor (instance explanation): %s' % (' AND '.join(explanation.anchor)))
    st.write('Precision: %.2f' % explanation.precision)
    st.write('Coverage: %.2f' % explanation.coverage)
# st.write("""### Trust score:""")
    ts = TrustScore(k_filter=10,
                alpha=.05,
                filter_type='distance_knn',
                leaf_size=40,
                metric='euclidean',
                dist_filter_type='point')
    ts.fit(X_train, y_train, classes=len(class_names))
    score, closest_class = ts.score(X_test[idx-1].reshape(1,-1),
                                y_pred[idx-1], k=2,  # kth nearest neighbor used
                                              # to compute distances for each class
                                dist_type='point')  # 'point' or 'mean' distance option
    st.write('Trust score: {}'.format(score))
    st.write('\nClosest not predicted class: {}'.format(closest_class))
    # dataset_choice = st.sidebar.selectbox('Select a dataset:', ('Iris', 'Wine', 'Cancer'))
    target_classes = {0:'Setosa', 1:'Versicolor', 2:'Virginica'}
    if dataset_choice == 'Wine':
        selected_dataset = load_wine()
        selected_dataset_df = load_wine(as_frame=True)
    elif dataset_choice == 'Cancer':
        selected_dataset = load_breast_cancer()
        selected_dataset_df = load_breast_cancer(as_frame=True)
    else:
        st.write('0: Setosa')
        st.write('1: Versicolor')
        st.write('2: Virginica')
        # st.write(target_classes[closest_class])
