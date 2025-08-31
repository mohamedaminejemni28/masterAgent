

# === SETUP ===
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import os, json
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain.document_loaders import CSVLoader
from typing import List
from langchain.schema import BaseMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
import datetime
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
import re


# Prompt pour les questions générales
prompt_chatbot = PromptTemplate.from_template("""
Vous êtes un assistant intelligent et concis. Répondez uniquement à la question posée en utilisant des connaissances générales, sans inclure de contexte ou d'instructions dans la réponse.  

**Règles** :  
1. Répondez de manière directe, concise et professionnelle.  
2. Si la question est vague ou ne peut être répondue avec des connaissances générales, répondez : **"Rien de particulier, comment puis-je vous aider aujourd'hui ?"**  
3. Ne fabriquez pas d'informations.  

**Question** : {question}  
**Réponse** :  
""")

prompt_suivi = PromptTemplate.from_template("""
Vous êtes un assistant intelligent de suivi de projets. Répondez uniquement à la question posée en utilisant les données du contexte fourni, sans inclure le contexte ou les instructions dans la réponse.  

**Contexte** (non affiché dans la réponse) :  
{context}  

**Colonnes des données** :  
- **ID_Projet** : identifiant unique (ex. : PRJ-0001)  
- **Client** : nom du client  
- **Description** : résumé du projet  
- **Statut** : 'En cours', 'Finalisé', 'Pas commencé', etc.  
- **Date_Début** : date de début  
- **Jours_écoulés** : jours depuis la Hawkins de la date de début  
- **État** : avancement ou blocage (ex. : 'à risque', 'retard', 'normal')  
- **Inactivité** : indique si le projet est inactif ('non' si 'Finalisé')  
- **Action_suggérée** : action recommandée (ex. : 'relancer le client')  
- **Type_Action** : type d’action (ex. : alerte, e-mail, réunion)  

**Règles** :  
1. Pour une liste de projets (ex. : "donner les projets"), produisez un tableau Markdown avec toutes les colonnes : ID_Projet, Client, Description, Statut, Date_Début, Jours_écoulés, État, Inactivité, Action_suggérée, Type_Action.  
2. Pour une réponse unique (ex. : question sur un projet précis par ID), formulez une phrase concise et professionnelle.  
3. Pour une question impliquant une action ou alerte, basez-vous sur **Inactivité**, **État**, **Action_suggérée**, et **Type_Action**.  
4. Si aucune donnée pertinente, répondez : **"Aucune donnée disponible dans le tableau."**  
5. Si la question est hors contexte, répondez : **"Je ne sais pas."**  
6. Exemple pour une liste de projets :  
   **Question** : Donner les projets avec Statut en cours  
   **Réponse** :  
   | ID_Projet | Client | Description | Statut | Date_Début | Jours_écoulés | État | Inactivité | Action_suggérée | Type_Action |  
   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |  
   | PRJ-0001 | Client_1 | Projet de service 1 | En cours | 2024-06-01 | 439 | En traitement | Oui | Relancer le client | Urgente |  

**Question** : {question}  
**Réponse** :  
""")



prompt_formation = PromptTemplate.from_template("""
Tu es un assistant intelligent spécialisé en analyse des formations en ligne et concurrentielles.

Chaque document contient des informations telles que :
- Formation
- Title
- Note
- URL
- Date de mise à jour / Offre

Instructions :
- Fournis la réponse uniquement sous forme de liste claire avec : **Titre, Plateforme, Note, Lien**.
- Trie les résultats si nécessaire (par date, par note, par popularité).
- Si aucune donnée ne correspond, réponds : *"Aucune formation trouvée correspondant à votre demande."*
- N'inclus ni le contexte ni les instructions dans la réponse.

Documents disponibles :
{context}

Question de l’utilisateur : {question}
""")




from langchain.prompts import PromptTemplate



def load_suivi(file_path_excel, sheet_name="Suivi_Projets", filtre_statut=None, output_csv="data.csv"):
    aujourd_hui = pd.Timestamp(datetime.datetime.now().date())

    def suggérer_action(row):
        if row['Statut'] == 'Finalisé':
            return "Aucune action"
        elif row['Jours_écoulés'] > 30:
            return "Relancer le client"
        elif 15 < row['Jours_écoulés'] <= 40:
            return "Suivi en cours"
        elif row['Jours_écoulés'] <= 15:
            return "Attente de réponse client"
        else:
            return "Analyser situation"

    def calcul_etat(row):
        if row['Statut'] == "Finalisé":
            return "Terminé"
        elif row['Jours_écoulés'] < 0:
            return "Pas encore débuté"
        else:
            return "En traitement"

    # Lecture du fichier Excel
    df_projets = pd.read_excel(file_path_excel, sheet_name=sheet_name)

    # Correction éventuelle du nom de colonne
    if "Date_D�but" in df_projets.columns:
        df_projets.rename(columns={"Date_D�but": "Date_Début"}, inplace=True)

    # Conversion des dates
    df_projets['Date_Début'] = pd.to_datetime(df_projets['Date_Début'], errors='coerce')

    # Application éventuelle du filtre sur le statut
    if filtre_statut:
        df = df_projets[df_projets['Statut'].str.lower() == filtre_statut.lower()].copy()
    else:
        df = df_projets.copy()

    # Calcul des colonnes analytiques
    df['Jours_écoulés'] = (aujourd_hui - df['Date_Début']).dt.days
    df['Etat'] = df.apply(calcul_etat, axis=1)
    df['Inactivité'] = df.apply(
        lambda row: "Non" if row['Statut'] == "Finalisé" else ("Oui" if row['Jours_écoulés'] > 350 else "Non"),
        axis=1
    )
    df['Action_suggérée'] = df.apply(suggérer_action, axis=1)
    df['Type_Action'] = df['Action_suggérée'].map({
        "Aucune action": "Aucune",
        "Relancer le client": "Urgente",
        "Suivi en cours": "Normale",
        "Attente de réponse client": "Faible",
        "Analyser situation": "Modérée"
    })
    df['Email'] = df['Client'].apply(lambda x: x.lower().replace(" ", "").replace("é", "e") + "@gmail.com")

    # Modification spécifique demandée
    df.loc[df["Client"] == "Client_2", "Email"] = "aminejemni181@gmail.com"

    # Sauvegarde CSV
    df.to_csv(output_csv, index=False, encoding="utf-8")

    # Chargement dans LangChain
    loader = CSVLoader(file_path=output_csv, encoding="utf-8")
    docs = loader.load()

    return df, docs







# Exemple d’utilisation :
import os 
import sys
import os
import pandas as pd
import datetime


df_analyse, docs1 = load_suivi("Agent_Commercial_IA_SFM_Gigantic_Dataset.xlsx", sheet_name="Suivi_Projets")















import pandas as pd
from datetime import datetime
import re # Nécessaire pour l'extraction de fréquence
import os


# Exemple d’utilisation :
import os 
import sys
import os
import pandas as pd
import datetime




print("Début du script data_analyzer.py")
DATA_EXCEL_FILE = 'Agent_Commercial_IA_SFM_Gigantic_Dataset.xlsx' 
DATA_EXCEL_SHEET_NAME = 'Tendances_Interets'


# In[ ]:





# In[105]:


def get_data_for_trend_analysis(num_rows: int = 100, sector: str = None) -> list[dict]:
    """
    Lit les données pertinentes du fichier Excel pour l'analyse de tendance.
    Retourne les dernières 'num_rows' entrées sous forme de liste de dictionnaires,
    en respectant une limite maximale de lignes pour éviter le dépassement de contexte.
    Accepte maintenant un paramètre optionnel 'sector' pour filtrer les données.
    """
    print(f"Appel de la fonction get_data_for_trend_analysis() pour {num_rows} lignes...")

    max_rows_limit = 20 
    
    try:
        df = pd.read_excel(DATA_EXCEL_FILE, sheet_name=DATA_EXCEL_SHEET_NAME)
        print(f"Fichier Excel '{DATA_EXCEL_FILE}' lu avec succès depuis la feuille '{DATA_EXCEL_SHEET_NAME}'.")

        relevant_columns = ['ID_Client', 'Secteur', 'Produit_Préféré', 'Fréquence_Demande']

        current_relevant_columns = []
        for col in relevant_columns:
            if col in df.columns:
                current_relevant_columns.append(col)
            else:
                print(f"Avertissement: La colonne '{col}' est manquante dans le fichier Excel. Elle sera ignorée.")
        
        if not current_relevant_columns:
            print("Erreur: Aucune colonne pertinente trouvée pour l'analyse de tendance.")
            return []

        df_filtered = df[current_relevant_columns].copy()

        # --- NOUVELLE LOGIQUE DE FILTRAGE PAR SECTEUR ---
        if sector and 'Secteur' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['Secteur'].str.contains(sector, case=False, na=False)]
            print(f"Données filtrées pour le secteur : '{sector}'.")

        if 'Fréquence_Demande' in df_filtered.columns:
            def extract_frequency(freq_str):
                if isinstance(freq_str, str):
                    match = re.search(r'(\d+)', freq_str)
                    return int(match.group(1)) if match else 0
                return freq_str

            df_filtered['Fréquence_Demande_Num'] = df_filtered['Fréquence_Demande'].apply(extract_frequency)
            df_filtered = df_filtered.sort_values(by='Fréquence_Demande_Num', ascending=False)
            df_filtered = df_filtered.drop(columns=['Fréquence_Demande_Num'])

        num_rows_to_return = min(num_rows, len(df_filtered), max_rows_limit)
        data_subset = df_filtered.sample(n=num_rows_to_return, random_state=42)

        if data_subset.empty:
            print(f"Aucune donnée pertinente trouvée pour l'analyse de tendance. Vérifiez le secteur spécifié ou les données.")
            return []

        data_list = data_subset.to_dict(orient='records')
        print(f"Nombre de lignes de données préparées pour l'analyse de tendance : {len(data_list)}")
        return data_list

    except FileNotFoundError:
        print(f"Erreur: Le fichier '{DATA_EXCEL_FILE}' n'a pas été trouvé. Assurez-vous qu'il est dans le même répertoire que le script.")
        return []
    except ValueError as ve:
        print(f"Erreur: Problème lors de la lecture du fichier Excel ou de la feuille '{DATA_EXCEL_SHEET_NAME}'. Détails: {ve}")
        return []
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors de la lecture des données pour l'analyse de tendance : {e}")
        return []

if __name__ == "__main__":
    print("\n--- Test direct de data_analyzer.py ---")
    test_data = {
        'ID_Client': [f'CLI-{i:04d}' for i in range(1, 101)],
        'Secteur': ['Santé', 'Finance', 'Éducation', 'Industrie', 'IT'] * 20,
        'Produit_Préféré': ['IA', 'Cloud', 'Sécurité', 'ERP', 'Data Analytics'] * 20,
        'Fréquence_Demande': [f'{i % 5 + 1} fois par mois' for i in range(100)]
    }
    temp_df = pd.DataFrame(test_data)
    temp_excel_file = "Temp_Trend_Data.xlsx"
    temp_sheet_name = "Analyse_Tendances"
    temp_df.to_excel(temp_excel_file, sheet_name=temp_sheet_name, index=False)
    print(f"Fichier de test '{temp_excel_file}' créé avec {len(temp_df)} lignes.")
    original_excel_file = DATA_EXCEL_FILE
    original_sheet_name = DATA_EXCEL_SHEET_NAME
    DATA_EXCEL_FILE = temp_excel_file
    DATA_EXCEL_SHEET_NAME = temp_sheet_name
    
    print("\nTest avec un secteur spécifique 'Finance':")
    sample_data_finance = get_data_for_trend_analysis(num_rows=50, sector='Finance')
    if sample_data_finance:
        print(f"Exemple de données pour 'Finance' : {len(sample_data_finance)} lignes.")

    DATA_EXCEL_FILE = original_excel_file
    DATA_EXCEL_SHEET_NAME = original_sheet_name
    if os.path.exists(temp_excel_file):
        os.remove(temp_excel_file)
        print(f"\nFichier de test '{temp_excel_file}' supprimé.")

    print("--- Fin du test direct ---")


# In[106]:


# Importations des bibliothèques nécessaires
import os
import re
import ast
from langchain.agents import AgentExecutor, initialize_agent
from langchain.tools import StructuredTool 
from langchain_community.llms import HuggingFacePipeline

from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain_google_genai import ChatGoogleGenerativeAI

print("Début du script trend_analysis_agent.py")

# --- Fonction d'initialisation de l'agent ---
def initialize_trend_analysis_agent(llm_instance: ChatGoogleGenerativeAI) -> AgentExecutor:
    """
    Initialise et configure l'agent d'analyse de tendance.
    Cette fonction retourne l'AgentExecutor, qui peut être appelé par l'orchestrateur.
    """
    
    print("Outils de l'agent d'analyse de tendance chargés.")
    
    # --- 2. Définition des Outils pour l'Agent avec StructuredTool ---
    # Utilisez StructuredTool.from_function pour gérer plusieurs arguments
    tools = [
        StructuredTool.from_function(
            func=get_data_for_trend_analysis,
            name="get_data_for_trend_analysis",
            description="""
            Utilisez cet outil pour récupérer les dernières lignes de données pertinentes
            pour l'analyse de tendance à partir du fichier Excel. Ces données incluent
            'ID_Client', 'Secteur', 'Produit_Préféré' et 'Fréquence_Demande'.
            Vous pouvez spécifier 'num_rows' (int) pour limiter le nombre de lignes (par défaut 100),
            et 'sector' (string) pour filtrer les données par secteur.
            L'outil retourne une liste de dictionnaires, chaque dictionnaire étant une ligne de données.
            Exemple d'utilisation: get_data_for_trend_analysis(num_rows=50, sector='Finance')
            """
        ),
    ]
    
    # --- 3. Création de l'Agent ---
    agent_prompt_content = """
    Tu es un agent d'analyse de tendance et un expert en données clients.
    Ton objectif est d'analyser les données de clients qui te sont fournies pour identifier des tendances,
    des motifs, et des informations clés sur les préférences des clients, les produits et les secteurs.
    Les données incluent 'ID_Client', 'Secteur', 'Produit_Préféré' et 'Fréquence_Demande'.
    **Tâche principale et Flux d'exécution CRUCIAL :**
    1. **DÈS LE DÉBUT, tu DOIS ABSOLUMENT utiliser l'outil `get_data_for_trend_analysis` UNIQUEMENT AVEC `num_rows=20` et, si nécessaire, un paramètre `sector` pour récupérer les 20 lignes de données les plus pertinentes.** C'est la première et la plus importante étape.
    2. **APRES avoir reçu les données via l'Observation**, analyse ces 20 lignes pour identifier :
        * Les produits les plus/moins populaires.
        * Les secteurs avec la plus forte/faible fréquence de demande.
        * Les corrélations entre secteurs et produits préférés.
        * Toute autre observation pertinente.
    3. **Fournis un résumé clair et concis** sous forme d'analyse structurée.
    Voici les outils que tu peux utiliser:
    {tools}
    Utilise le format suivant pour tes interactions:
    Question: la question d'entrée que tu dois analyser
    Thought: Je dois d'abord comprendre la question et déterminer si je dois utiliser un outil. Si oui, je décrirai mon plan pour l'outil. Sinon, je décrirai comment je vais répondre directement.
    Action:
    ```json
    {{
      "action": "nom_outil",
      "action_input": {{ "parametre1": "valeur1", "parametre2": "valeur2" }}
    }}
    Observation: le résultat de l'action
    ... (ce Thought/Action/Observation peut être répété plusieurs fois)
    Thought: J'ai effectué les actions nécessaires et j'ai toutes les informations. Je peux maintenant formuler ma réponse finale.
    Final Answer: La réponse finale doit être une analyse structurée des tendances sous forme de texte, avec des points clés.
    Commence ta réponse avec le "Thought:".
    **Exemple de première interaction :**
    Question: Analyse les données clients pour identifier les tendances.
    Thought: Mon objectif principal est d'analyser les données clients. La première étape cruciale est de récupérer les données en utilisant get_data_for_trend_analysis avec num_rows=20 comme spécifié dans ma tâche principale.
    Action:
    ```json
    {{
      "action": "get_data_for_trend_analysis",
      "action_input": {{"num_rows": 20}}
    }}
    Observation:
    """
    
    # Initialisation de l'agent
    agent_executor = initialize_agent(
        tools,
        llm_instance,
        agent="structured-chat-zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "prompt": PromptTemplate(template=agent_prompt_content, input_variables=["tools", "input", "agent_scratchpad"])
        }
    )
    print("Agent d'analyse de tendance créé avec le type ReAct et AgentExecutor configuré.")
    return agent_executor





# In[108]:




# In[109]:




from relances import *
from concurrent import docs3,prompt_concurrent

from formation import *

from avis_marche import *
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Création de l'embedding
embedding_model2 = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Création du vector store avec FAISS
vector_store1 = FAISS.from_documents(
    documents=docs1,
    embedding=embedding_model2
)




from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Création de l'embedding
embedding_model2 = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Création du vector store avec FAISS
vector_store2 = FAISS.from_documents(
    documents=docs2,
    embedding=embedding_model2
)


# In[83]:


from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Création de l'embedding
embedding_model2 = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Création du vector store avec FAISS
vector_store3 = FAISS.from_documents(
    documents=docs3,
    embedding=embedding_model2
)


# In[ ]:








from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyBEY6b4GSKOV_qeaAjmzoyNEVU5y2tcsQA"

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  
    temperature=0.0
)





import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
EMAIL_SENDER = "aminejemni181@gmail.com"
EMAIL_PASSWORD = "zjaa qwfo dtet uonb" 

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def envoyer_email_relance(client, email, description, id_projet):
    sujet = f"Relance projet {id_projet}"
    corps = f"""Bonjour {client},

Nous souhaitons faire un point concernant le projet **{id_projet}** :
« {description} »

Merci de nous tenir informés de son avancement ou de nous proposer un créneau pour en discuter.

Cordialement,  
L’équipe de SFM technologies"""

    message = MIMEMultipart()
    message["From"] = EMAIL_SENDER
    message["To"] = email
    message["Subject"] = sujet
    message.attach(MIMEText(corps, "plain"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(message)
            print(f"Email envoyé à {client} ({email}) pour le projet {id_projet}")
    except Exception as e:
        print(f"Erreur pour {client} ({email}) : {e}")





# In[111]:


import re
import pandas as pd

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[_\-]", " ", text)  
    text = re.sub(r"[^\w\s]", "", text)  
    text = re.sub(r"\s+", " ", text)  
    return text.strip()

def executer_actions_suivi(question, df_analyse):
    keywords = ["envoie", "email", "mail", "relance", "envoyer"]
    question_lower = question.lower()
    
    if any(keyword in question_lower for keyword in keywords):
        client_cible = None
        id_projet_cible = None

        match_client = re.search(r"client[_\s\-]*([a-zA-Z0-9]+)", question, re.IGNORECASE)
        if match_client:
            client_cible = match_client.group(0).strip() 

        match_projet = re.search(r"projet[_\s\-]*([a-zA-Z0-9]+)", question, re.IGNORECASE)
        if match_projet:
            id_projet_cible = match_projet.group(0).strip()

        projets_a_relancer = df_analyse[df_analyse["Action_suggérée"].str.lower() == "relancer le client"]
        if client_cible:
            client_cible_norm = normalize_text(client_cible)
            projets_a_relancer = projets_a_relancer[
                projets_a_relancer["Client"].apply(normalize_text) == client_cible_norm
            ]

        if id_projet_cible:
            id_projet_cible_norm = normalize_text(id_projet_cible)
            projets_a_relancer = projets_a_relancer[
                projets_a_relancer["ID_Projet"].apply(normalize_text) == id_projet_cible_norm
            ]

        if projets_a_relancer.empty:
            return "Aucun projet à relancer correspondant à la demande."

        for _, ligne in projets_a_relancer.iterrows():
            client = ligne["Client"]
            email = ligne.get("Email", "")
            if pd.notna(email) and email.strip() != "":
                envoyer_email_relance(client, email, ligne["Description"], ligne["ID_Projet"])
            else:
                print(f" Email manquant pour {client}, aucun envoi effectué.")

        return f"{len(projets_a_relancer)} e-mails de relance envoyés."

    return None


# In[112]:


import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
import re
import dateparser

df = pd.read_csv("data.csv")

def extraire_date_heure_manuel(texte):
    jours_semaine = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    pattern = r"le\s+(" + "|".join(jours_semaine) + r")\s+(\d{1,2})\s+([a-zA-Z]+)(?:\s+(\d{4}))?\s+à\s+(\d{1,2})h"
    match = re.search(pattern, texte, re.IGNORECASE)
    if not match:
        return None

    jour_num = int(match.group(2))
    mois_str = match.group(3).lower()
    annee_str = match.group(4)
    heure = int(match.group(5))

    mois_map = {
        "janvier": "janvier", "fevrier": "février", "mars": "mars", "avril": "avril",
        "mai": "mai", "juin": "juin", "juillet": "juillet", "aout": "août",
        "septembre": "septembre", "octobre": "octobre", "novembre": "novembre", "decembre": "décembre",
    }
    mois_str = mois_map.get(mois_str, mois_str)
    annee = int(annee_str) if annee_str else datetime.now().year

    date_str = f"{jour_num} {mois_str} {annee} {heure}:00"
    date = dateparser.parse(date_str, languages=["fr", "en"])
    return date

def traiter_demande(user_input):
    if "rendez-vous" in user_input.lower() or "rdv" in user_input.lower() or "reunion" in user_input.lower():
        match = re.search(r"projet\s*(\d+)", user_input.lower())
        if not match:
            return " Aucun ID de projet trouvé dans la phrase."

        numero_projet = int(match.group(1))
        id_proj_complet = f"PRJ-{numero_projet:04d}"

        projet = df[df["ID_Projet"] == id_proj_complet]
        if projet.empty:
            return f" Le projet {id_proj_complet} est introuvable."

        date_rdv = extraire_date_heure_manuel(user_input)
        if not date_rdv:
            return " Aucune date/heure reconnue dans la phrase."

        ligne = projet.iloc[0]
        client_nom = ligne["Client"]
        client_email = ligne["Email"]
        description = ligne["Description"]

        summary = f"RDV avec {client_nom} (Projet {id_proj_complet})"
        description_event = description
        fixed_tz = timezone(timedelta(hours=1))
        start_time = date_rdv.replace(tzinfo=fixed_tz)
        end_time = start_time + timedelta(minutes=30)

        start_iso = start_time.isoformat()
        end_iso = end_time.isoformat()

        payload = {
            "calendarId": "aminejemni181@gmail.com",
            "summary": summary,
            "description": description_event,
            "start": {"dateTime": start_iso, "timeZone": "Africa/Tunis"},
            "end": {"dateTime": end_iso, "timeZone": "Africa/Tunis"},
            "attendees": [{"email": client_email}],
            "conferenceData": {
                "createRequest": {
                    "requestId": f"req-{datetime.now().timestamp()}",
                    "conferenceSolutionKey": {"type": "hangoutsMeet"}
                }
            }
        }

        webhook_url = "https://mohamed228.app.n8n.cloud/webhook/google calendar"  

        try:
            response = requests.post(webhook_url, json=payload)
            if response.status_code == 200:
                jour_tu = date_rdv.strftime('%A %d %B %Y').capitalize()
                heure_tu = date_rdv.strftime('%Hh%M')
                return f"RDV créé avec succès pour {client_nom} le {jour_tu} à {heure_tu} !"
            else:
                return f" Erreur lors de l'envoi à n8n : {response.status_code}\n{response.text}"
        except Exception as e:
            return f" Exception lors de l'appel au webhook : {e}"

    return None


# In[113]:


class ResearchState(Dict):
    messages: List[Dict]
    next: Optional[str]
    cycle: int


# In[ ]:





# In[114]:


def create_chatbot_agent(prompt, llm, agent_name="Chatbot Agent"):
    def agent(messages: List[Dict]) -> Dict:
        question = messages[-1]["content"] if messages else ""
        full_prompt = prompt.format(question=question)
        response = llm.invoke(full_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        if "Réponse :" in response_text:
            idx = response_text.index("Réponse :") + len("Réponse :")
            cleaned = response_text[idx:].strip()
        else:
            cleaned = response_text.strip()
        return {"content": cleaned}
    return agent




def create_suivi_agent(prompt, vector_store, llm, agent_name="Suivi Agent"):
    if vector_store is None:
        return lambda messages: {"content": "Aucune donnée disponible dans le tableau."}

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def format_context(docs):
        unique_docs = list(dict.fromkeys([doc.page_content for doc in docs]))
        return "\n\n".join(unique_docs)

    rag_chain = (
        RunnableMap({
            "context": lambda x: format_context(retriever.invoke(x["question"])),
            "question": lambda x: x["question"]
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    def agent(messages: List[Dict]) -> Dict:
        question = messages[-1]["content"] if messages else ""
        question_lower = question.lower()
        keywords_relance = ["envoie","envoi", "email", "relance", "envoyer"]
        keywords_rdv = ["rendez-vous", "reunion", "rdv"]

        # 🔹 Vérifier si c'est une action "relance email"
        if any(keyword in question_lower for keyword in keywords_relance):
            print("-> Détection d'une demande d'envoi de relance, exécution...")
            response = executer_actions_suivi(question, df_analyse)
            print("Résultat action:", response)
            return {"content": response}

        # 🔹 Vérifier si c'est une action "prise de rendez-vous"
        elif any(keyword in question_lower for keyword in keywords_rdv):
            print("-> Détection d'une demande de rendez-vous, exécution...")
            response = traiter_demande(question)
            print("Résultat action:", response)
            return {"content": response}

        # 🔹 Sinon, exécution classique (RAG)
        else:
            response = rag_chain.invoke({"question": question})
            if "**Réponse** :" in response:
                idx = response.index("**Réponse** :") + len("**Réponse** :")
                cleaned = response[idx:].strip()
            else:
                cleaned = response.strip()

            if cleaned.startswith("- ") and "donner les projet" in question_lower:
                cleaned = "Aucune donnée disponible dans le tableau. La réponse doit être un tableau Markdown."

            return {"content": cleaned}

    return agent


# In[116]:


# Agent pour les questions liées aux formations
def create_formation_agent(prompt, vector_store, llm, agent_name="Formation Agent"):
    if vector_store is None:
        return lambda messages: {"content": "Aucune donnée disponible dans le tableau."}
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    def format_context(docs):
        unique_docs = list(dict.fromkeys([doc.page_content for doc in docs]))
        return "\n\n".join(unique_docs)

    
    rag_chain = (
        RunnableMap({
            "context": lambda x: format_context(retriever.invoke(x["question"])),
            "question": lambda x: x["question"]
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    def validate_date(date_str):
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    def validate_row(row, question):
        parts = row.split("|")
        if len(parts) < 11:
            return False
        date_debut = parts[5].strip()
        plateforme = parts[11].strip() if len(parts) > 11 else ""
        # Vérifier le format de la date
        if not validate_date(date_debut):
            return False
        # Vérifier la plateforme si spécifiée
        if "edx" in question.lower() and plateforme.lower() != "edx":
            return False
        return True
    def agent(messages: List[Dict]) -> Dict:
        question = messages[-1]["content"] if messages else ""
        response = rag_chain.invoke({"question": question})
        if "**Réponse** :" in response:
            idx = response.index("**Réponse** :") + len("**Réponse** :")
            cleaned = response[idx:].strip()
        else:
            cleaned = response.strip()
        # Filtrer et valider les données
        if cleaned.startswith("|"):
            lines = cleaned.split("\n")
            header = lines[0]
            separator = lines[1]
            data_rows = [line for line in lines[2:] if validate_row(line, question)]
            if not data_rows:
                return {"content": "Aucune donnée disponible dans le tableau pour la question posée."}
            # Limiter à deux formations si demandé
            if "donner deux formations" in question.lower():
                data_rows = data_rows[:2]
            cleaned = "\n".join([header, separator] + data_rows)
        # Vérifier si le format est incorrect
        if cleaned.startswith("- ") and "donner les formation" in question.lower():
            cleaned = "Aucune donnée disponible dans le tableau. La réponse doit être un tableau Markdown."
        return {"content": cleaned}
    return agent





from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict


# Agent pour les questions liées aux concurrents
def create_concurrent_agent(prompt, vector_store, llm, agent_name="Concurrent Agent"):
    if vector_store is None:
        return lambda messages: {"content": "Aucune donnée disponible pour les concurrents."}

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Fonction pour formater le contexte (concaténation des contenus des docs)
    def format_context(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # Définition de la chaîne RAG
    rag_chain = (
        RunnableMap({
            "context": lambda x: format_context(retriever.invoke(x["question"])),
            "question": lambda x: x["question"]
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    # Fonction principale de l’agent
    def agent(messages: List[Dict]) -> Dict:
        question = messages[-1]["content"] if messages else ""
        response = rag_chain.invoke({"question": question})

        # Nettoyage de la réponse si nécessaire
        if "**Réponse** :" in response:
            idx = response.index("**Réponse** :") + len("**Réponse** :")
            cleaned = response[idx:].strip()
        else:
            cleaned = response.strip()

        return {"content": cleaned}

    return agent

# In[133]:


from typing import List, Dict

def create_coordinator_agent(llm, agent_name="Coordinator Agent"):
    def agent(messages: List[Dict]) -> Dict:
        question = messages[-1]["content"].lower() if messages else ""
        project_keywords = [
            "projet", "project", "id_projet", "client", "description", "statut",
            "date_début", "jours_écoulés", "état", "inactivité", "action_suggérée",
            "type_action", "suivi", "en cours", "finalisé", "pas commencé", "à risque",
            "retard", "normal", "alerte", "e-mail", "réunion"
        ]
        formation_keywords = [
            "edx", "datacamp", "udemy", "coursera", "formation"
        ]
        concurrent_keywords = [
            "concurrent", "concurrents", "id_concurrent", "concurent",
            "produit", "service", "tarif", "nouveau_tarif",
            "offre spéciale", "compétiteur", "compétition"
        ]
        trend_keywords = [
            "tendance", "trend", "analyse des tendances", "préférences clients"
        ]
        market_avis_keywords = [

        ]
        relance_keywords = [
            "relance", "prospect", "whatsapp", "appel", "visio",
            "relancer", "contact", "suivi client", "relance commerciale",
            "message de relance", "planification", "téléphone", "visioconférence",
            "statut prospect", "action relance"
        ]
        # Logique de routage
        if any(keyword in question for keyword in market_avis_keywords):
            next_node = "market_avis"
        elif any(keyword in question for keyword in relance_keywords):
            next_node = "relance"
        elif any(keyword in question for keyword in trend_keywords):
            next_node = "trend"
        elif any(keyword in question for keyword in project_keywords) and "suivi" in question:
            next_node = "suivi"
        elif any(keyword in question for keyword in formation_keywords):
            next_node = "formation"
        elif any(keyword in question for keyword in concurrent_keywords):
            next_node = "concurrent"
        elif any(keyword in question for keyword in project_keywords):
            next_node = "suivi"
        else:
            next_node = "chatbot"
        print(f"\n--- Coordinator Agent: Routing '{question}' to node '{next_node}' ---\n")
        return {"content": f'{{"next": "{next_node}"}}'}
    return agent





from typing import Dict, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI

def relance_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node for processing prospect relance actions based on user input keywords.
   
    Args:
        state: Dictionary containing the current state, including messages with user input.
   
    Returns:
        Updated state with the results of relance actions and next step.
    """
    # Extract user input (choice) from the last message
    question = state["messages"][-1]["content"].strip().lower()
   
    # Initialize the language model and relance agent
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0
    )
    tools = initialize_relance_agent(llm)
   
    # List of possible action types
    action_types = ['email', 'whatsapp', 'appel', 'visio']
   
    # Initialize response message
    response_content = []
   
    # Check for keywords in the user input or special case '6'
    selected_actions = []
    if any(keyword in question for keyword in ['mail', 'email', 'emails']):
        selected_actions = ['email']
        response_content.append("Traitement de l'action : email")
    elif any(keyword in question for keyword in ['whatsapp', 'whats app', 'whats']):
        selected_actions = ['whatsapp']
        response_content.append("Traitement de l'action : whatsapp")
    elif any(keyword in question for keyword in ['appel', 'appels', 'call', 'calls']):
        selected_actions = ['appel']
        response_content.append("Traitement de l'action : appel")
    elif any(keyword in question for keyword in ['visio', 'video', 'visioconférence']):
        selected_actions = ['visio']
        response_content.append("Traitement de l'action : visio")
    elif question == '6':
        response_content.append("Arrêt du programme.")
        return {
            **state,
            "messages": state["messages"] + [{"content": response_content[0]}],
            "next": "output"
        }
    else:
        response_content.append("Choix invalide. Veuillez indiquer une action (mail/email, whatsapp, appel/call, visio/video) ou 6 pour arrêter.")
        return {
            **state,
            "messages": state["messages"] + [{"content": response_content[0]}],
            "next": "output"
        }
   
    for action_type in selected_actions:
        response_content.append(f"\nTraitement des prospects pour l'action : {action_type}")
       
        prospects = read_prospects_for_action(action_type)
       
        if isinstance(prospects, str):
            response_content.append(prospects)  # Error or no prospects message
            continue
       
        # Process each prospect
        for prospect in prospects:
            prospect_id = prospect['ID_Relance']
            client_name = prospect['Client']
            email = prospect['Email']
            telephone = prospect['Telephone']
            action = prospect['Prochaine_Action'].lower()
            status = prospect['Statut']
            last_relance_date = prospect['Date_Dernière_Relance']
           
            response_content.append(f"\nTraitement du prospect ID: {prospect_id} ({client_name})")
            response_content.append(f"Statut: {status}, Dernière relance: {last_relance_date}")
           
            try:
                # Execute the appropriate action
                if action == 'email':
                    result = send_relance_email._run(
                        email_address=email,
                        subject=f"Relance commerciale pour {client_name}",
                        body=f"Bonjour {client_name},\n\nCeci est un message de relance. Merci de nous contacter pour discuter de votre projet.\nCordialement,\nL'équipe commerciale"
                    )
                    response_content.append(result)
               
                elif action == 'whatsapp':
                    result = send_whatsapp_message._run(
                        telephone_number=telephone,
                        message=f"Bonjour {client_name}, ceci est un message de relance. Contactez-nous pour plus d'informations !"
                    )
                    response_content.append(result)
               
                elif action == 'appel':
                    result = make_phone_call._run(
                        telephone_number=telephone,
                        reason=f"Relance commerciale pour {client_name}"
                    )
                    response_content.append(result)
               
                elif action == 'visio':
                    result = schedule_visio_call._run(
                        email_address=email,
                        subject=f"Planification d'une visioconférence avec {client_name}"
                    )
                    response_content.append(result)
               
                else:
                    response_content.append(f"Action inconnue : {action}")
                    continue
               
                # Update prospect status after successful relance
                update_result = mark_prospect_as_done._run(
                    prospect_id=prospect_id,
                    action=f"{action_type.capitalize()} effectué"
                )
                response_content.append(update_result)
               
            except Exception as e:
                response_content.append(f"Erreur lors du traitement du prospect {prospect_id}: {e}")
   
    # Combine all responses into a single message
    response = {"content": "\n".join(response_content)}
   
    # Update and return the state
    return {
        **state,
        "messages": state["messages"] + [response],
        "next": "output"
    }





def avis_marche_node(state: ResearchState) -> ResearchState:

    question = state["messages"][-1]["content"]
    
    # Initialisation de l'agent de veille des marchés publics
    market_agent = initialize_market_watch_agent(llm_gemini)
    
    try:
        agent_response = market_agent.invoke({"input": question})
    except Exception as e:
        agent_response = {"content": f"Erreur lors de la recherche des avis de marchés : {e}"}
    
    # S'assurer que la réponse est un dictionnaire avec 'content'
    if isinstance(agent_response, dict) and "output" in agent_response:
        response = {"content": agent_response["output"]}
    else:
        response = {"content": str(agent_response)}

    # Mise à jour de l'état
    return {
        **state,
        "messages": state["messages"] + [response],
        "next": "output"
    }




def trend_node(state: ResearchState) -> ResearchState:
    question = state["messages"][-1]["content"]
    trend_agent = initialize_trend_analysis_agent(llm_gemini)
    agent_response = trend_agent.invoke({"input": question})

    # Ensure the response is a dict with 'content'
    if isinstance(agent_response, dict) and "content" in agent_response:
        response = agent_response
    else:
        response = {"content": str(agent_response)}

    return {
        **state,
        "messages": state["messages"] + [response],
        "next": "output"
    }




def coordinator_node(state: ResearchState) -> ResearchState:
    coordinator = create_coordinator_agent(llm_gemini, agent_name="Coordinator Agent")
    response = coordinator(state["messages"])
    next_node = eval(response["content"])["next"]
    return {
        **state,
        "next": next_node
    }




def chatbot_node(state: ResearchState) -> ResearchState:
    question = state["messages"][-1]["content"]


    chatbot_agent = create_chatbot_agent(prompt_chatbot, llm_gemini, agent_name="Chatbot Agent")
    response = chatbot_agent([{"content": question}])
    return {
        **state,
        "messages": state["messages"] + [response],
        "next": "output"
    }




def suivi_node(state: ResearchState) -> ResearchState:
    question = state["messages"][-1]["content"]
    suivi_agent = create_suivi_agent(prompt_suivi, vector_store1, llm_gemini, agent_name="Suivi Agent")
    response = suivi_agent([{"content": question}])
    return {
        **state,
        "messages": state["messages"] + [response],
        "next": "output"
    }




def formation_node(state: ResearchState) -> ResearchState:
    question = state["messages"][-1]["content"]
    formation_agent = create_formation_agent(prompt_formation, vector_store2, llm_gemini, agent_name="Formation Agent")
    response = formation_agent([{"content": question}])
    return {
        **state,
        "messages": state["messages"] + [response],
        "next": "output"
    }




def concurrent_node(state: ResearchState) -> ResearchState:
    question = state["messages"][-1]["content"]
    concurrent_agent = create_concurrent_agent(
        prompt_concurrent, vector_store3, llm_gemini, agent_name="Concurrent Agent"
    )
    response = concurrent_agent([{"role": "user", "content": question}])
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": response["content"]}],
        "next": "output"
    }





def output_node(state: ResearchState) -> ResearchState:
    print("\n--- Output Node: Returning final result ---")
    return state



from langgraph.graph import StateGraph, START, END
from typing import Dict, Any

def build_dynamic_multi_agent_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("suivi", suivi_node)
    workflow.add_node("formation", formation_node)
    workflow.add_node("concurrent", concurrent_node)

    workflow.add_node("output", output_node)
    workflow.add_edge(START, "coordinator")
    workflow.add_conditional_edges(
        "coordinator",
        lambda state: state["next"],
        {
            "chatbot": "chatbot",
            "suivi": "suivi",
            "formation": "formation",
            "concurrent": "concurrent",
            "output": "output"
        }
    )
    workflow.add_edge("chatbot", "output")
    workflow.add_edge("suivi", "output")
    workflow.add_edge("formation", "output")
    workflow.add_edge("concurrent", "output")
    workflow.add_edge("output", END)
    return workflow.compile()









def agent_response(question: str):
    compiled_graph = build_dynamic_multi_agent_graph()
    initial_state = {
        "messages": [{"content": question}],
        "next": "",
        "cycle": 0
    }

    final_state = compiled_graph.invoke(initial_state)

    if "messages" in final_state and final_state["messages"]:
        response_content = final_state["messages"][-1]["content"]

        # Nettoyage si "**Réponse** :" est présent
        if "**Réponse** :" in response_content:
            idx = response_content.index("**Réponse** :") + len("**Réponse** :")
            response_content = response_content[idx:].strip()

        # Vérification format incorrect pour projets, formations ou concurrents
        if response_content.startswith("- ") and (
            "donner les projets" in question.lower() 
            or "donner les formation" in question.lower()
        ):
            response_content = "Aucune donnée disponible dans le tableau. La réponse doit être un tableau Markdown."
    else:
        response_content = "Aucune réponse trouvée."

    return response_content




def clean_response(text):
    if "Réponse :" in text:
        return text.split("Réponse :")[-1].strip()
    return text.strip()




if __name__ == "__main__":
    while True:
        question = input("Pose la question : ")
        if question.lower() == "exit":
            print("Fin du programme.")
            break
        response = agent_response(question)
        s = clean_response(response)
        print("Réponse :", s)
        print("-" * 50)





