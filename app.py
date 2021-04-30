import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, precision_score
from sklearn.metrics import mean_squared_error

st.title('My first app using streamlit')

image = Image.open('datascience.png')
st.image(image, use_column_width=True)
st.write('This is a better interactive way to demonstrate the analysis to your collegues')





def main():
    option = st.sidebar.selectbox('Select your option', ('EDA', 'Visualisation', 'Building the model'))
    if option == 'EDA':
        st.subheader('Exploratory Data Analysis')
        data = st.file_uploader('Please upload your daatset here:',type=['csv','xlsx','txt','json'])
        st.success('File successfully uploaded')
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(50))
            
            if st.checkbox('Display Shape'):
                st.write(df.shape)
            if st.checkbox('Display categorical variables'):
                st.write(df.select_dtypes(include='object').head(50))
            if st.checkbox('Display numerical variables'):
                st.write(df.select_dtypes(exclude='object').head(50))
            if st.checkbox('Display Description'):
                st.write(df.describe().T)
            if st.checkbox('Select Multiple Colmuns'):
                selected_columns = st.multiselect('Select your prefered columns. NB: Make sure your final selected colmun is your target variable', df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1.head(50))
            if st.checkbox('Display Correlation'):
                st.write(df.corr())
            if st.checkbox('Correlation between mutitple variables'):
                corr_colmuns = st.multiselect('Select your prefered columns:', df.select_dtypes(exclude='object').columns)
                df_corr = df[corr_colmuns]
                st.write(df_corr.corr())
    
    if option == 'Visualisation':
        st.subheader('Visualisation')
        data = st.file_uploader('Please upload your data set', ('csv','xlsx','txt','json'))
        st.success('File successfully uploaded')
        if data is not None :
            df = pd.read_csv(data)
            st.dataframe(df.head(50))
            
            if st.checkbox('Select Multiple columns to plot'):
                selected_columns=st.multiselect('Select your preferred columns', df.columns)
                df1=df[selected_columns]
                st.dataframe(df1.head(50))
            if st.checkbox('Display Heatmap'):
                fig, ax = plt.subplots(figsize=(18,14))
                st.write(sns.heatmap(df.corr(), annot=True, cmap='viridis', ax=ax))
                st.pyplot(fig)
            if st.checkbox('Display Distplot'):
                box_columns = st.selectbox('Please select your prefered column', df.columns)
                fig1, ax1 = plt.subplots(figsize=(18,14))
                st.write(sns.distplot(df[box_columns]))
                st.pyplot(fig1)
            if st.checkbox('Display Pairplot'):
                st.write(sns.pairplot(df1,diag_kind='kde'))
                st.pyplot()
            if st.checkbox('Display Pie Chart'):
                all_columns = df.columns.to_list()
                pie_columns=st.selectbox("Select olumn to display", all_columns)
                pieChart = df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pieChart)
                st.pyplot()
    
    if option == 'Building the model':
        st.subheader('BUilding the model')
        data = st.file_uploader('Please upload your preferred data. NB: be sure your data contains only numeric variables',type=['csv','txt','xlxs','json'])
        st.success('File successfully uploaded')
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(50))
            if st.checkbox('Select Multiple Columns'):
                new_data=st.multiselect("Select your preferred columns. NB:Let your target variable be the last column", df.columns)
                df1=df[new_data]
                st.dataframe(df1.head(10))
                # Dividing my data into x and y variables
                X=df1.iloc[:,0:-1]
                y=df1.iloc[:,-1]
            seed=st.sidebar.slider('Seed',1,200)
            classifier_name = st.sidebar.selectbox('Select your model', ('SVM','Logistic Regression','Decision Tree','KNN','Naive Bayes')) 
            
            def add_parameter(name_of_clf):
                params = dict()
                if name_of_clf =='SVM':
                    c_svm=st.sidebar.slider('C', 0.01, 15.0)
                    params['C_SVM']=c_svm
                if name_of_clf == 'KNN':
                    k=st.sidebar.slider('K', 1, 15)
                    params['K']=k
                if name_of_clf == 'Logistic Regression':
                    c_lr =st.sidebar.slider('C', 0.1, 10.0)
                    params['C_LR']=c_lr
                if name_of_clf == 'Decision Tree':
                    max_depth=st.sidebar.slider('Max depth', 1, 30)
                    params['max_depth']=max_depth
                return params
            param=add_parameter(classifier_name)

            def get_classifier(name_of_clf, params):
                clf=None
                if name_of_clf == 'SVM':
                    clf = SVC(C=params['C_SVM'])
                if name_of_clf == 'KNN':
                    clf = KNeighborsClassifier(n_neighbors=params['K'])                
                if name_of_clf == 'Logistic Regression':
                    clf =  LogisticRegression(C=params['C_LR'])
                if name_of_clf == 'Decision Tree':
                    clf = DecisionTreeClassifier(max_depth=params['max_depth'])
                if name_of_clf == 'Naive Bayes':
                    clf = GaussianNB()
                return clf
            
            clf = get_classifier(classifier_name, param)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision  = precision_score(y_test, y_pred)
            st.write('Name of classifier:', classifier_name)
            st.write('Accuracy', accuracy)
            st.write('Precision', precision)
            st.write('Classification Report',classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()
