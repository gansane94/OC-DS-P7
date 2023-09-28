import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import shap
import requests
import plotly.graph_objects as go

# URL de l'API
API_URL = "https://votre-url-d-api.herokuapp.com/"

# Chargement des données
data_train = pd.read_csv('train_df_sample.csv')
data_test = pd.read_csv('test_df_sample.csv')

# Fonctions

def preprocessing(df, scaler):
    """Prétraitement des données avec le scaler spécifié.
    :param: df, scaler (str).
    :return: df_scaled.
    """
    cols = df.select_dtypes(['float64']).columns
    df_scaled = df.copy()
    if scaler == 'minmax':
        scal = MinMaxScaler()
    else:
        scal = StandardScaler()

    df_scaled[cols] = scal.fit_transform(df[cols])
    return df_scaled

def get_prediction(client_id):
    """Obtient la probabilité de défaut du client via l'API.
    :param: client_id (int).
    :return: probabilité de défaut (float) et décision (str)
    """
    url_get_pred = API_URL + "prediction/" + str(client_id)
    response = requests.get(url_get_pred)
    proba_default = round(float(response.content), 3)
    best_threshold = 0.54
    decision = "Refusé" if proba_default >= best_threshold else "Accordé"

    return proba_default, decision

def plot_score_gauge(proba):
    """Affiche une jauge indiquant le score du client.
    :param: proba (float).
    """
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=proba * 100,
        mode="gauge+number+delta",
        title={'text': "Jauge de score"},
        delta={'reference': 54},
        gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
               'bar': {'color': "MidnightBlue"},
               'steps': [
                   {'range': [0, 20], 'color': "Green"},
                   {'range': [20, 45], 'color': "LimeGreen"},
                   {'range': [45, 54], 'color': "Orange"},
                   {'range': [54, 100], 'color': "Red"}],
               'threshold': {'line': {'color': "brown", 'width': 4}, 'thickness': 1, 'value': 54}}))

    st.plotly_chart(fig)

def main():
    # Titre de la page
    st.set_page_config(page_title="Dashboard Prêt à dépenser", layout="wide")

    # Sidebar
    with st.sidebar:
        logo = Image.open('img/logo pret à dépenser.png')
        st.image(logo, width=200)
        # Page selection
        page = st.selectbox('Navigation', ["Home", "Information du client", "Interprétation locale",
                                           "Interprétation globale"])

        # ID Selection
        st.markdown("""---""")

        list_id_client = list(data_test['SK_ID_CURR'])
        list_id_client.insert(0, '<Select>')
        id_client_dash = st.selectbox("ID Client", list_id_client)
        st.write('Vous avez choisi le client ID : '+str(id_client_dash))

        st.markdown("""---""")
        st.write("Created by Océane Youyoutte")

    if page == "Home":
        st.title("Dashboard Prêt à dépenser - Home Page")
        st.markdown("Ce site contient un dashboard interactif permettant d'expliquer aux clients les raisons\n"
                    "d'approbation ou refus de leur demande de crédit.\n"

                    "\nLes prédictions sont calculées à partir d'un algorithme d'apprentissage automatique, "
                    "préalablement entraîné. Il s'agit d'un modèle *Light GBM* (Light Gradient Boosting Machine). "
                    "Les données utilisées sont disponibles [ici](https://www.kaggle.com/c/home-credit-default-risk/data). "
                    "Lors du déploiement, un échantillon de ces données a été utilisé.\n"

                    "\nLe dashboard est composé de plusieurs pages :\n"
                    "- **Information du client**: Vous pouvez y retrouver toutes les informations relatives au client "
                    "selectionné dans la colonne de gauche, ainsi que le résultat de sa demande de crédit. "
                    "Je vous invite à accéder à cette page afin de commencer.\n"
                    "- **Interprétation locale**: Vous pouvez y retrouver quelles caractéristiques du client ont le plus "
                    "influencé le choix d'approbation ou refus de la demande de crédit.\n"
                    "- **Intérprétation globale**: Vous pouvez y retrouver notamment des comparaisons du client avec "
                    "les autres clients de la base de données ainsi qu'avec des clients similaires.")

    if page == "Information du client":
        st.title("Dashboard Prêt à dépenser - Page Information du client")

        st.write("Cliquez sur le bouton ci-dessous pour commencer l'analyse de la demande :")
        button_start = st.button("Statut de la demande")
        if button_start:
            if id_client_dash != '<Select>':
                # Calcul des prédictions et affichage des résultats
                st.markdown("RÉSULTAT DE LA DEMANDE")
                probability, decision = get_prediction(id_client_dash)

                if decision == 'Accordé':
                    st.success("Crédit accordé")
                else:
                    st.error("Crédit refusé")

                # Affichage de la jauge
                plot_score_gauge(probability)

        # Affichage des informations client
        with st.expander("Afficher les informations du client", expanded=False):
            st.info("Voici les informations du client:")
            st.write(pd.DataFrame(data_test.loc[data_test['SK_ID_CURR'] == id_client_dash]))

    if page == "Interprétation locale":
        st.title("Dashboard Prêt à dépenser - Page Interprétation locale")

        locale = st.checkbox("Interprétation locale")
        if locale:
            st.info("Interprétation locale de la prédiction")
            shap_val = get_shap_val_local(id_client_dash)
            nb_features = st.slider('Nombre de variables à visualiser', 0, 20, 10)
            # Affichage du waterfall plot : shap local
            fig = shap.waterfall_plot(shap_val, max_display=nb_features, show=False)
            st.pyplot(fig)

            with st.expander("Explication du graphique", expanded=False):
                st.caption("Ici sont affichées les caractéristiques influençant de manière locale la décision. "
                           "C'est-à-dire que ce sont les caractéristiques qui ont influençé la décision pour ce client "
                           "en particulier.")

    if page == "Interprétation globale":
        st.title("Dashboard Prêt à dépenser - Page Interprétation globale")
        # Création du dataframe de voisins similaires
        data_voisins = df_voisins(id_client_dash)

        globale = st.checkbox("Importance globale")
        if globale:
            st.info("Importance globale")
            shap_values = get_shap_val()
            data_test_std = preprocessing(data_test.drop('SK_ID_CURR', axis=1), 'std')
            nb_features = st.slider('Nombre de variables à visualiser', 0, 20, 10)
            fig, ax = plt.subplots()
            # Affichage du summary plot : shap global
            ax = shap.summary_plot(shap_values[1], data_test_std, plot_type='bar', max_display=nb_features)
            st.pyplot(fig)

            with st.expander("Explication du graphique", expanded=False):
                st.caption("Ici sont affichées les caractéristiques influençant de manière globale la décision.")

        distrib = st.checkbox("Comparaison des distributions")
        if distrib:
            st.info("Comparaison des distributions de plusieurs variables de l'ensemble de données")
            # Possibilité de choisir de comparer le client sur l'ensemble de données ou sur un groupe de clients similaires
            distrib_compa = st.radio("Choisissez un type de comparaison :", ('Tous', 'Clients similaires'), key='distrib')

            list_features = list(data_train.columns)
            list_features.remove('SK_ID_CURR')
            # Affichage des distributions des variables renseignées
            with st.spinner(text="Chargement des graphiques..."):
                col1, col2 = st.columns(2)
                with col1:
                    feature1 = st.selectbox("Choisissez une caractéristique", list_features,
                                            index=list_features.index('AMT_CREDIT'))
                    if distrib_compa == 'Tous':
                        distribution(feature1, id_client_dash, data_train)
                    else:
                        distribution(feature1, id_client_dash, data_voisins)
                with col2:
                    feature2 = st.selectbox("Choisissez une caractéristique", list_features,
                                            index=list_features.index('EXT_SOURCE_2'))
                    if distrib_compa == 'Tous':
                        distribution(feature2, id_client_dash, data_train)
                    else:
                        distribution(feature2, id_client_dash, data_voisins)

                with st.expander("Explication des distributions", expanded=False):
                    st.caption("Vous pouvez sélectionner la caractéristique dont vous souhaitez observer la distribution. "
                               "En bleu est affichée la distribution des clients qui ne sont pas considérés en défaut et "
                               "dont le prêt est donc jugé comme accordé. En orange, à l'inverse, est affichée la "
                               "distribution des clients considérés comme faisant défaut et dont le prêt leur est refusé. "
                               "La ligne pointillée verte indique où se situe le client par rapport aux autres clients.")

        bivar = st.checkbox("Analyse bivariée")
        if bivar:
            st.info("Analyse bivariée")
            # Possibilité de choisir de comparer le client sur l'ensemble de données ou sur un groupe de clients similaires
            bivar_compa = st.radio("Choisissez un type de comparaison :", ('Tous', 'Clients similaires'), key='bivar')

            list_features = list(data_train.columns)
            list_features.remove('SK_ID_CURR')
            list_features.insert(0, '<Select>')

            # Selection des features à afficher
            c1, c2 = st.columns(2)
            with c1:
                feat1 = st.selectbox("Sélectionner une caractéristique X ", list_features)
            with c2:
                feat2 = st.selectbox("Sélectionner une caractéristique Y", list_features)
            # Affichage des nuages de points de la feature 2 en fonction de la feature 1
            if (feat1 != '<Select>') & (feat2 != '<Select>'):
                if bivar_compa == 'Tous':
                    scatter(id_client_dash, feat1, feat2, data_train)
                else:
                    scatter(id_client_dash, feat1, feat2, data_voisins)
                with st.expander("Explication des scatter plot", expanded=False):
                    st.caption("Vous pouvez ici afficher une caractéristique en fonction d'une autre. "
                               "En bleu sont indiqués les clients ne faisant pas défaut et dont le prêt est jugé comme "
                               "accordé. En rouge, sont indiqués les clients faisant défaut et dont le prêt est jugé "
                               "comme refusé. L'étoile noire correspond au client et permet donc de le situer par rapport "
                               "à la base de données clients.")

        boxplot = st.checkbox("Analyse des boxplot")
        if boxplot:
            st.info("Comparaison des distributions de plusieurs variables de l'ensemble de données à l'aide de boxplot.")

            feat_quanti = data_train.select_dtypes(['float64']).columns
            # Selection des features à afficher
            features = st.multiselect("Sélectionnez les caractéristiques à visualiser: ",
                                      sorted(feat_quanti),
                                      default=['AMT_CREDIT', 'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3'])

            # Affichage des boxplot
            boxplot_graph(id_client_dash, features, data_voisins)
            with st.expander("Explication des boxplot", expanded=False):
                st.caption("Les boxplot permettent d'observer les distributions des variables renseignées. "
                           "Une étoile violette représente le client. Ses plus proches voisins sont également "
                           "renseignés sous forme de points de couleurs (rouge pour ceux étant qualifiés comme "
                           "étant en défaut et vert pour les autres).")

if __name__ == "__main__":
    main()
