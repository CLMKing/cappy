import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import f1_score, classification_report, confusion_matrix, plot_confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import confusion_matrix, balanced_accuracy_score
# from sklearn.svm import SVC
# from sklearn.preprocessing import MinMaxScaler
# from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC
# from sklearn.model_selection import KFold, StratifiedKFold
# from xgboost import XGBClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import RadiusNeighborsClassifier

# from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier, BalancedRandomForestClassifier
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler

# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.multiclass import OutputCodeClassifier
# from sklearn.multiclass import OneVsOneClassifier

import statistics


#####

st.set_page_config('wide')


def make_recomm(test_sample, max_recomm, div, trained_model):
    limit = 0
    count_pre = 0
    count_post = 0
    recomm_pre = []
    recomm_post = []
    recomm_dict = {}

    # st.write('------')
    # st.subheader('Current Mileage')
    # st.write(f"CURRENT MILEAGE: {test_sample['Mileage In']} + '\t'")
    increment = int(test_sample['Mileage In'] / div)

    test_samp_post1 = test_sample.copy()
    test_samp_pre1 = test_sample.copy()

    while (count_pre < 5) and (limit < max_recomm):
        limit += 1
        test_samp_pre1['Mileage In'] -= increment

        if test_samp_pre1['Mileage In'] >= 0:
            pred = trained_model.predict(pd.DataFrame([test_samp_pre1.values], columns=test_samp_pre1.index))
            recomm_pre.append(lencoder.inverse_transform(pred)[0])
            count_pre = len(set(recomm_pre))
            # st.write(f"Mileage: {test_samp_pre1['Mileage In']}")
            # st.write('\t' + lencoder.inverse_transform(pred)[0] + '\n')
        else:
            break
    # st.write('END OF PRE RECOMMS\n\n\n')
    # st.write('------')
    limit = 0

    # st.write(f"CURRENT MILEAGE: {test_sample['Mileage In']}\n")
    while (count_post < 5) and (limit < max_recomm):
        limit += 1
        test_samp_post1['Mileage In'] += increment

        if True:
            pred = trained_model.predict(pd.DataFrame([test_samp_post1.values], columns=test_samp_post1.index))
            recomm_post.append(lencoder.inverse_transform(pred)[0])
            count_post = len(set(recomm_post))
            # st.write(f"Mileage: {test_samp_post1['Mileage In']}")
            # st.write(lencoder.inverse_transform(pred)[0] + '\n')
    # st.write('END OF POST RECOMMS\n\n\n')

    # st.write(f"Length of Pre Recommendations: {len(recomm_pre)}")
    # st.write(f"No. of distinct Pre Recommendations: {count_pre}")
    # st.write(f"Length of Post Recommendations: {len(recomm_post)}")
    # st.write(f"No. of distinct Post Recommendations: {count_post}")

    # recomm_dict['pre'] = list(set(recomm_pre))
    # recomm_dict['post'] = list(set(recomm_post))

    st.subheader('Pre Recommended Service Packages')
    st.write(list(set(recomm_pre)))
    st.subheader('Post Recommended Service Packages')
    st.write(list(set(recomm_post)))

#######################

st.sidebar.header('Eskwelabs X Autoserve')

# st.sidebar.markdown("![Alt Text](https://media4.giphy.com/media/coxQHKASG60HrHtvkt/200.gif)", width = 100)
st.sidebar.image('https://i.pinimg.com/originals/4f/97/1b/4f971b0d6bacdd50c85333a2af80ddaf.gif', width = 250)
st.sidebar.subheader('Navigation')
st.sidebar.write('------')
nav = st.sidebar.radio("", ['Model I', 'Model II'])

if nav == 'Model I':
    # core code model 1 #
    final_df = pd.read_csv('FINAL_FINAL.csv')

    new_df = final_df[
        final_df['Service Category'].isin(final_df['Service Category'].value_counts().nsmallest(9).index.values)]

    X = new_df[['Make', 'Year', 'Model', 'Mileage In']]
    Y = new_df['Service Category']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=13)

    lencoder = LabelEncoder()
    lencoder.fit(Y_train)

    Y_train_lencoded = lencoder.transform(Y_train)
    Y_test_lencoded = lencoder.transform(Y_test)

    encoder = ce.TargetEncoder(cols=['Make', 'Year', 'Model'])
    encoder.fit(X_train, Y_train_lencoded)

    X_train_encoded = encoder.transform(X_train)
    X_test_encoded = encoder.transform(X_test)

    # getting the pickle model
    model_1 = pickle.load(open('sample_1.sav', 'rb'))

    # fitting the pickle model
    model_1.fit(X_train_encoded, Y_train_lencoded)
    y_pred = model_1.predict(X_test_encoded)

    # metrics
    acc = model_1.score(X_test_encoded, Y_test_lencoded)
    balanced_acc = balanced_accuracy_score(Y_test_lencoded, y_pred)
    f1_micro = f1_score(Y_test_lencoded, y_pred, average='micro')
    f1_macro = f1_score(Y_test_lencoded, y_pred, average='macro')
    target_names = lencoder.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8])
    class_report = classification_report(Y_test_lencoded, y_pred, target_names=target_names)

    # confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    np.set_printoptions(precision=2)
    plt.rcParams.update({'font.size': 10})
    # plt.xlabel('', fontsize = 25)
    # plt.ylabel('', fontsize = 25)
    # plt.title('', fontsize = 30)
    title = 'Model 1 - Confusion Matrix (Recall)'
    disp = plot_confusion_matrix(model_1, X_test_encoded, Y_test_lencoded,
                                 display_labels=target_names,
                                 cmap=plt.cm.Blues, ax=ax, xticks_rotation=90)
    disp.ax_.set_title(title)

    #####################
    st.title("Model Test Run")
    if st.checkbox('Model I Metrics'):
        st.subheader('Model I Classification Report')
        st.write('------')
        st.subheader('Classification Report')
        st.text(class_report)
        # st.write(f'Accuracy: {round(acc,3)}')
        # st.write(f'Balanced accuracy: {round(balanced_acc, 3)}')
        # st.write(f'f-score_micro: {round(f1_micro,3)}')
        # st.write(f'f-score_macro: {round(f1_macro,3)}')
        # st.write('------')
        st.subheader('Confusion Matrix')
        st.write('------')
        st.pyplot(fig)
        st.write('------')

    st.subheader('Please enter sample index')
    input_1 = st.number_input('Index number')
    input_1 = int(input_1)
    st.write('------')
    st.header('Recommended Result')
    # locating the test sample via index
    test_sample = X_test_encoded.iloc[input_1]
    make_recomm(test_sample, 200, 20, model_1)
    st.success('Loading Sucessful!')

if nav == 'Model II':
    # core code #
    final_df = pd.read_csv('FINAL_FINAL.csv')
    new_df = final_df[
        final_df['Service Category'].isin(final_df['Service Category'].value_counts().nlargest(8).index.values)]

    X = new_df[['Make', 'Year', 'Model', 'Mileage In']]
    Y = new_df['Service Category']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=13)
    lencoder = LabelEncoder()
    lencoder.fit(Y_train)

    Y_train_lencoded = lencoder.transform(Y_train)
    Y_test_lencoded = lencoder.transform(Y_test)

    encoder = ce.TargetEncoder(cols=['Make', 'Year', 'Model'])
    encoder.fit(X_train, Y_train_lencoded)

    X_train_encoded = encoder.transform(X_train)
    X_test_encoded = encoder.transform(X_test)

    # getting the pickle model
    model_2 = pickle.load(open('sample_2.sav', 'rb'))

    # fitting the pickle model
    model_2.fit(X_train_encoded, Y_train_lencoded)
    y_pred = model_2.predict(X_test_encoded)

    # metrics
    acc = model_2.score(X_test_encoded, Y_test_lencoded)
    balanced_acc = balanced_accuracy_score(Y_test_lencoded, y_pred)
    f1_micro = f1_score(Y_test_lencoded, y_pred, average='micro')
    f1_macro = f1_score(Y_test_lencoded, y_pred, average='macro')
    target_names = lencoder.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7])
    class_report = classification_report(Y_test_lencoded, y_pred, target_names=target_names)

    # confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    np.set_printoptions(precision=2)
    plt.rcParams.update({'font.size': 12})
    # plt.xlabel('', fontsize = 25)
    # plt.ylabel('', fontsize = 25)
    # plt.title('', fontsize = 30)
    title = 'Model II - Confusion Matrix (Recall)'
    disp = plot_confusion_matrix(model_2, X_test_encoded, Y_test_lencoded,
                                 display_labels=target_names,
                                 cmap=plt.cm.Blues, ax=ax, xticks_rotation=90)
    disp.ax_.set_title(title)


    #############
    st.title("Model Test Run")
    if st.checkbox('Model II Metrics'):
        st.title('Model II Classification Report')
        st.write('------')
        st.subheader('Classification Report')
        st.text(class_report)
        # st.write(f'Accuracy: {round(acc, 3)}')
        # st.write(f'Balanced accuracy: {round(balanced_acc, 3)}')
        # st.write(f'f-score_micro: {round(f1_micro, 3)}')
        # st.write(f'f-score_macro: {round(f1_macro, 3)}')
        st.write('------')
        st.subheader('Confusion Matrix')
        st.pyplot(fig)

    st.subheader('Please enter sample index')
    input_2 = st.number_input('Index number')
    input_2 = int(input_2)
    st.write('------')
    st.header('Recommended Result')
    # locating the test sample via index
    test_sample = X_test_encoded.iloc[input_2]
    make_recomm(test_sample, 200, 20, model_2)
    st.success('Loading Sucessful!')








