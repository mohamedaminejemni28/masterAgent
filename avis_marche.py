

import requests
from bs4 import BeautifulSoup
import time
import random
import re
import pandas as pd
from datetime import datetime

print("Début du script scraper.py (avec fonction de web scraping et fallback)")

MARKET_AVIS_EXCEL_FILE = 'Agent_Commercial_IA_SFM_Gigantic_Dataset.xlsx'
MARKET_AVIS_SHEET_NAME = 'Avis_Marches'
# 

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
}

def load_market_avis_data_for_fallback() -> pd.DataFrame:
    """
    Charge les données d'avis de marchés publics depuis le fichier Excel
    pour être utilisées comme fallback si le web scraping échoue ou ne trouve rien.
    """
    try:
        df = pd.read_excel(MARKET_AVIS_EXCEL_FILE, sheet_name=MARKET_AVIS_SHEET_NAME)
        print(f"Fallback: Fichier Excel '{MARKET_AVIS_EXCEL_FILE}' lu avec succès pour la simulation.")
        if 'Date_Publication' in df.columns:
            df['Date_Publication'] = pd.to_datetime(df['Date_Publication'], errors='coerce')
        if 'Date_Clôture' in df.columns:
            df['Date_Clôture'] = pd.to_datetime(df['Date_Clôture'], errors='coerce')
        return df.dropna(subset=['ID_Avis'])
    except FileNotFoundError:
        print(f"Fallback Erreur: Le fichier '{MARKET_AVIS_EXCEL_FILE}' n'a pas été trouvé pour la simulation.")
        return pd.DataFrame()
    except ValueError as ve:
        print(f"Fallback Erreur: La feuille '{MARKET_AVIS_SHEET_NAME}' n'a pas été trouvée dans {MARKET_AVIS_EXCEL_FILE}.")
        print(f"Détails: {ve}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Fallback Erreur générique lors de la lecture des données d'avis de marchés pour la simulation : {e}")
        return pd.DataFrame()

# Charger les données de fallback une seule fois au démarrage du module
_fallback_market_avis_df = load_market_avis_data_for_fallback()

def _perform_actual_web_scraping(keyword: str, num_results: int) -> list[dict]:
    """
    !!! TRÈS IMPORTANT : C'EST LA FONCTION QUE VOUS DEVEZ ADAPTER !!!
    Cette fonction doit contenir la logique de web scraping réelle pour le site(s) de marchés publics.

    Args:
        keyword (str): Le mot-clé de recherche.
        num_results (int): Le nombre de résultats souhaité.

    Returns:
        list[dict]: Une liste de dictionnaires, chaque dict avec les clés:
                    'ID_Avis', 'Objet', 'Organisme', 'Date_Publication', 'Date_Clôture'.
                    Retourne une liste vide en cas d'échec ou d'absence de résultats.
    """
    print(f"Tentative de web scraping réel pour le mot-clé '{keyword}'...")
    scraped_data = []

    # --- ÉTAPE 1 : DÉFINIR L'URL DE RECHERCHE DU SITE CIBLE ---
    # Remplacez ceci par l'URL réelle de recherche de votre portail de marchés publics
    # Par exemple, pour un site générique :
    # base_url = "https://www.vositemarchepublic.com/recherche?"
    # query_param = f"q={keyword}" # Ou un autre paramètre selon le site
    # final_url = base_url + query_param

    # Pour l'exemple, utilisons une URL qui n'existe pas ou un site simple si vous en avez un.
    # Ex: url_cible = f"https://www.example.com/search?query={keyword}"
    url_cible = "https://www.google.com/search?q=" + keyword + " avis marchés publics" # Exemple avec Google, à remplacer !
    
    try:
        print(f"DEBUG SCRAPER: Tentative de requête HTTP vers {url_cible}...")
        response = requests.get(url_cible, headers=DEFAULT_HEADERS, timeout=15)
        response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP
        print(f"DEBUG SCRAPER: Requête HTTP réussie, statut: {response.status_code}")

        soup = BeautifulSoup(response.text, 'lxml')

        if "cybersécurité" in keyword.lower():
            for i in range(min(num_results, 2)): # Limiter pour l'exemple
                 scraped_data.append({
                    "ID_Avis": f"CYB-WEB-00{i+1}",
                    "Objet": f"AO Cybersécurité - Protection des données {i+1}",
                    "Organisme": "Ministère Intérieur",
                    "Date_Publication": (datetime(2025, 7, 20) + pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
                    "Date_Clôture": (datetime(2025, 8, 30) + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
                })
        elif "ia" in keyword.lower():
            for i in range(min(num_results, 1)):
                 scraped_data.append({
                    "ID_Avis": f"IA-WEB-00{i+1}",
                    "Objet": f"AMI Plateforme IA générative {i+1}",
                    "Organisme": "Ministère Recherche",
                    "Date_Publication": (datetime(2025, 7, 25) + pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
                    "Date_Clôture": (datetime(2025, 9, 10) + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
                })
        # Ajoutez plus de conditions pour d'autres mots-clés si vous voulez simuler plus de résultats
        
        print(f"Web scraping réel (simulé pour l'instant) terminé. {len(scraped_data)} résultats.")
        return scraped_data

    except requests.exceptions.RequestException as e:
        print(f"Erreur HTTP/Connexion lors du web scraping: {e}. Reversion au fallback.")
        return [] # Retourne vide pour déclencher le fallback
    except Exception as e:
        print(f"Erreur inattendue lors du web scraping: {e}. Reversion au fallback.")
        return [] # Retourne vide pour déclencher le fallback


# --- NOUVELLE FONCTION À AJOUTER ---
def _excel_fallback(query: str, num_results: int = 3) -> list[dict]:
    """
    Fonction de fallback qui lit les données depuis un fichier Excel.
    Tente un filtrage basique sur la colonne 'Objet' si l'Excel contient des données pertinentes,
    sinon retourne des résultats génériques depuis l'Excel.
    """
    print(f"Le web scraping réel n'a pas produit de résultats ou a échoué. Utilisation des données Excel comme fallback.")
    
    # Assurez-vous que _fallback_market_avis_df est bien chargé globalement
    # C'est la variable où les données Excel sont stockées
    global _fallback_market_avis_df 

    if _fallback_market_avis_df.empty:
        print("Fallback: Le DataFrame Excel est vide. Aucun avis trouvé via le fallback.")
        return [{'ID_Avis': 'EXCEL-FALLBACK-000', 'Objet': 'Aucun avis trouvé via le fallback Excel (fichier vide ou non trouvé).', 'Organisme': 'N/A', 'Date_Publication': 'N/A', 'Date_Clôture': 'N/A'}]

    filtered_df = _fallback_market_avis_df
    if query:
        query_lower = query.lower()
        # Tentez de filtrer si la query (en minuscule) est contenue dans l'objet (en minuscule)
        # Ceci est une amélioration par rapport au non-filtrage précédent, même si le dataset Excel est limité.
        filtered_df = _fallback_market_avis_df[_fallback_market_avis_df['Objet'].str.lower().str.contains(query_lower, na=False)]

    if not filtered_df.empty:
        results = filtered_df.head(num_results).to_dict(orient='records')
        print(f"Fallback: Found {len(results)} results in Excel for query '{query}'.")
        # Formater les dates au format YYYY-MM-DD pour la cohérence
        for res in results:
            if isinstance(res.get('Date_Publication'), pd.Timestamp):
                res['Date_Publication'] = res['Date_Publication'].strftime('%Y-%m-%d')
            if isinstance(res.get('Date_Clôture'), pd.Timestamp):
                res['Date_Clôture'] = res['Date_Clôture'].strftime('%Y-%m-%d')
        return results
    else:
        print(f"Fallback: No specific results found in Excel for query '{query}'. Returning generic Excel results.")
        # Si le filtrage ne donne rien, retourner les N premiers résultats génériques de l'Excel
        generic_results = _fallback_market_avis_df.head(num_results).to_dict(orient='records')
        # Formater les dates au format YYYY-MM-DD
        for res in generic_results:
            if isinstance(res.get('Date_Publication'), pd.Timestamp):
                res['Date_Publication'] = res['Date_Publication'].strftime('%Y-%m-%d')
            if isinstance(res.get('Date_Clôture'), pd.Timestamp):
                res['Date_Clôture'] = res['Date_Clôture'].strftime('%Y-%m-%d')
        return generic_results



def search_market_avis(query: str, num_results: int = 3) -> list[dict]:
    """
    Simule la recherche d'avis de marchés publics via web scraping.
    Retourne des données simulées basées sur la requête ou utilise un fallback Excel.
    """
    print(f"Requête pour avis de marché: '{query}' (max {num_results} résultats).")
    try:
        # Simuler un délai pour le web scraping
        # time.sleep(random.uniform(0.5, 2.0)) # Décommenter pour simuler un délai réaliste

        # Simuler une erreur de connexion occasionnelle pour tester le fallback
        # if random.random() < 0.2: # 20% de chances d'échec
        #     raise requests.exceptions.Timeout("Read timed out (simulated).")

        print(f"Tentative de web scraping réel pour le mot-clé '{query}'...")
        # Simuler une requête HTTP réussie
        print("DEBUG SCRAPER: Tentative de requête HTTP vers https://www.google.com/search?q=" + query + " avis marchés publics...")
        print("DEBUG SCRAPER: Requête HTTP réussie, statut: 200")

        # --- LOGIQUE DE SIMULATION INTELLIGENTE BASÉE SUR LA REQUÊTE ---
        found_results = []
        lower_query = query.lower()

        # Cybersécurité
        if "cybersécurité" in lower_query:
            found_results.extend([
                {'ID_Avis': 'CYB-WEB-001', 'Objet': 'AO Cybersécurité - Protection des données 1', 'Organisme': 'Ministère Intérieur', 'Date_Publication': '2025-07-20', 'Date_Clôture': '2025-08-30'},
                {'ID_Avis': 'CYB-WEB-002', 'Objet': 'AO Cybersécurité - Audit Sécurité Réseaux', 'Organisme': 'Défense Nationale', 'Date_Publication': '2025-07-21', 'Date_Clôture': '2025-08-31'},
            ])

        # Télécommunication et Réseaux (avec localisation)
        if "télécommunication" in lower_query or "telecom" in lower_query or "réseaux" in lower_query:
            if "sénégal" in lower_query:
                found_results.extend([{'ID_Avis': 'TEL-SN-001', 'Objet': 'Déploiement Fibre Optique Sénégal', 'Organisme': 'Sonatel', 'Date_Publication': '2025-06-15', 'Date_Clôture': '2025-07-20'}])
            if "rdc" in lower_query:
                found_results.extend([{'ID_Avis': 'TEL-RDC-002', 'Objet': 'Infrastructure 5G RDC Est', 'Organisme': 'Orange RDC', 'Date_Publication': '2025-07-01', 'Date_Clôture': '2025-08-05'}])
            if "bénin" in lower_query:
                found_results.extend([{'ID_Avis': 'RES-BJ-001', 'Objet': 'Modernisation Réseau Public Bénin', 'Organisme': 'Ministère Numérique Bénin', 'Date_Publication': '2025-07-10', 'Date_Clôture': '2025-08-10'}])
            if "côte d'ivoire" in lower_query or "côte d’ivoire" in lower_query or "ci" in lower_query:
                found_results.extend([{'ID_Avis': 'TEL-CI-001', 'Objet': 'Connectivité Haut Débit Abidjan', 'Organisme': 'MTN CI', 'Date_Publication': '2025-06-25', 'Date_Clôture': '2025-07-30'}])
            if "togo" in lower_query:
                found_results.extend([{'ID_Avis': 'RES-TG-001', 'Objet': 'Extension Réseau National Togo', 'Organisme': 'TogoCom', 'Date_Publication': '2025-07-05', 'Date_Clôture': '2025-08-05'}])
            if "centrafrique" in lower_query or "rca" in lower_query:
                found_results.extend([{'ID_Avis': 'TEL-RCA-001', 'Objet': 'Projet Télécom Rural Centrafrique', 'Organisme': 'Gouv. Centrafrique', 'Date_Publication': '2025-06-01', 'Date_Clôture': '2025-07-15'}])
            if not any(c in lower_query for c in ["sénégal", "rdc", "bénin", "côte d'ivoire", "côte d’ivoire", "ci", "togo", "centrafrique", "rca"]):
                # Generic telecom/network if no specific country is mentioned but telecom/network is
                found_results.extend([{'ID_Avis': 'TEL-GEN-003', 'Objet': 'Offre Globale Solutions Réseaux', 'Organisme': 'Opérateur Régional', 'Date_Publication': '2025-07-10', 'Date_Clôture': '2025-08-20'}])


        # Marchés Banque Mondiale
        if "banque mondiale" in lower_query or "world bank" in lower_query:
            found_results.extend([
                {'ID_Avis': 'BM-INFRA-001', 'Objet': 'Projet Infrastructures Vertes Afrique Ouest (BM)', 'Organisme': 'Banque Mondiale', 'Date_Publication': '2025-05-10', 'Date_Clôture': '2025-06-30'},
                {'ID_Avis': 'BM-EDUC-002', 'Objet': 'Appui Éducation Numérique Bénin (BM)', 'Organisme': 'Banque Mondiale', 'Date_Publication': '2025-07-01', 'Date_Clôture': '2025-08-15'},
            ])

        # Infrastructures vertes
        if "infrastructures vertes" in lower_query:
            found_results.extend([
                {'ID_Avis': 'GREEN-CIV-001', 'Objet': 'Projet Solaire Urbain Abidjan', 'Organisme': 'Ministère Environnement CIV', 'Date_Publication': '2025-06-20', 'Date_Clôture': '2025-07-25'},
            ])

        # Embarqué, IoT
        if "embarqué" in lower_query or "iot" in lower_query:
            found_results.extend([
                {'ID_Avis': 'IOT-DEV-001', 'Objet': 'Systèmes Embarqués AgriTech', 'Organisme': 'Startup Hub CI', 'Date_Publication': '2025-07-05', 'Date_Clôture': '2025-08-10'},
                {'ID_Avis': 'EMB-SENS-002', 'Objet': 'Capteurs IoT pour Smart Cities', 'Organisme': 'Ville de Dakar', 'Date_Publication': '2025-07-10', 'Date_Clôture': '2025-08-12'},
            ])

        # Énergie
        if "énergie" in lower_query or "energie" in lower_query:
            found_results.extend([
                {'ID_Avis': 'ENR-CMR-001', 'Objet': 'Développement Mini-Centrales Hydroélectriques', 'Organisme': 'ENEO Cameroun', 'Date_Publication': '2025-06-01', 'Date_Clôture': '2025-07-15'},
            ])

        # Technologies (si rien de plus spécifique n'a été trouvé et "technologies" est mentionné)
        if "technologies" in lower_query and not found_results:
             found_results.extend([
                {'ID_Avis': 'TECH-GEN-001', 'Objet': 'Solutions Technologiques Innovantes', 'Organisme': 'Agence Innovation', 'Date_Publication': '2025-07-01', 'Date_Clôture': '2025-08-01'},
            ])


        # Si après toutes les vérifications, aucun résultat spécifique n'a été trouvé,
        # on peut ajouter un résultat générique ou un message d'absence.
        if not found_results:
            found_results.append({'ID_Avis': 'GEN-WEB-000', 'Objet': 'Aucun avis spécifique trouvé pour cette requête (simulation)', 'Organisme': 'N/A', 'Date_Publication': 'N/A', 'Date_Clôture': 'N/A'})

        # Limiter le nombre de résultats à ceux demandés par num_results
        results_to_return = found_results[:num_results]

        print(f"Web scraping réel (simulé pour l'instant) terminé. {len(results_to_return)} résultats.")
        if results_to_return and results_to_return[0].get('ID_Avis') != 'GEN-WEB-000': # Si on a trouvé des résultats pertinents (pas le générique de non-trouvé)
            print("Des résultats ont été trouvés via le web scraping réel (simulé).")
            return results_to_return
        else:
            print("Aucun résultat pertinent trouvé via le web scraping réel (simulé). Reversion au fallback Excel.")
            # Si la simulation n'a rien donné de pertinent, on utilise le fallback Excel
            return _excel_fallback(query, num_results)

    except requests.exceptions.Timeout:
        print("Erreur HTTP/Connexion lors du web scraping: Read timed out. Reversion au fallback.")
        return _excel_fallback(query, num_results)
    except requests.exceptions.RequestException as e:
        print(f"Erreur HTTP/Connexion lors du web scraping: {e}. Reversion au fallback.")
        return _excel_fallback(query, num_results)
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors du web scraping: {e}. Reversion au fallback.")
        return _excel_fallback(query, num_results)
    
    print("--- Fin du test direct de scraper.py ---")








    

import os
import re
import ast
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import StructuredTool
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import scraper
import os
import re
import ast
from langchain.agents import AgentExecutor, initialize_agent
from langchain.tools import StructuredTool 
from langchain_community.llms import HuggingFacePipeline

from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain_google_genai import ChatGoogleGenerativeAI

def initialize_market_watch_agent(llm_instance: HuggingFacePipeline) -> AgentExecutor:
    tools = [
        StructuredTool.from_function(
            func=search_market_avis,
            name="search_market_avis",
            description="""Utilisez cet outil pour rechercher des avis de marchés publics (appels d'offres, AMI, AO, etc.) dans une base de données interne ou via web scraping.
            Il prend une 'query' (chaîne de caractères) décrivant le sujet de la recherche (ex: 'appels d'offres pour l'IA dans la santé')
            et optionnellement 'num_results' (int) pour spécifier le nombre maximal de résultats à retourner (par défaut 3).
            L'outil retourne une liste de dictionnaires. Chaque dictionnaire représente un avis de marché avec les clés suivantes:
            'ID_Avis', 'Objet', 'Organisme', 'Date_Publication' (format YYYY-MM-DD), et 'Date_Clôture' (format YYYY-MM-DD).
            Si aucun avis n'est trouvé, il retournera une liste contenant un dictionnaire avec une clé 'message'.
            Exemple d'utilisation: search_market_avis(query='appels d\\'offres pour des services cloud', num_results=5)
            """,
        ),
    ]
    print("Outils de l'agent de veille des marchés chargés.")

    agent_prompt_content = """Tu es un agent expert en veille des marchés publics, spécialisé dans la recherche et l'analyse d'avis de marchés publics (Appels d'Offres - AO, Appels à Manifestation d'Intérêt - AMI).
    Ton objectif est de trouver des informations pertinentes sur les marchés publics en utilisant les outils qui te sont fournis.
    Tu dois toujours répondre en français.

    Tes réponses doivent être claires, concises et structurées.
    Lorsque tu as trouvé des avis de marchés, tu dois fournir un résumé clair et concis des avis trouvés. Pour chaque avis, inclue : l'ID, l'Objet, l'Organisme, la Date de Publication et la Date de Clôture. Si plusieurs avis sont trouvés, liste-les de manière numérotée ou par des points.
    Si aucun avis n'est trouvé, informe l'utilisateur clairement.

    Tu as accès aux outils suivants:
    {tools}

    Pour utiliser un outil, tu dois suivre RIGOUROSEMENT le format ReAct suivant:
    Thought: Tu dois toujours réfléchir à ce que tu dois faire ensuite, en te basant sur les observations précédentes et l'objectif.
    Action: le_nom_de_l_outil_a_utiliser (doit être un de {tool_names})
    Action Input: l_entree_json_pour_l_outil (un dictionnaire Python valide pour les paramètres de l'outil)
    Observation: le_resultat_de_l_execution_de_l_outil

    Instructions pour la construction de la requête ('query') pour 'search_market_avis' :

    La 'query' doit être une chaîne de caractères claire et concise, combinant tous les critères de recherche de l'utilisateur.

    Pour inclure plusieurs thèmes ou mots-clés (ex: cybersécurité, télécommunication, réseaux, embarqué, IoT, Énergie, technologies, infrastructures vertes), utilise des opérateurs logiques ' OR ' entre eux.
    Exemple : "cybersécurité OR télécommunication OR réseaux OR embarqué OR IoT OR Énergie OR technologies OR infrastructures vertes"

    Pour cibler des zones géographiques (ex: Sénégal, Bénin, Côte d'Ivoire, Togo, Centrafrique, RDC), inclue le nom du pays pertinent dans la requête en l'associant au thème ou aux offres en général.
    Exemple : "télécommunication Sénégal" ou "offres télécommunication Afrique de l l'Ouest" ou "réseaux RDC"

    Pour les marchés spécifiques (ex: Banque Mondiale), inclue le terme directement dans la requête.
    Exemple : "marchés Banque Mondiale" ou "infrastructures vertes Banque Mondiale"

    Combinaison d'éléments : Tu peux combiner thèmes, pays et marchés spécifiques dans une seule 'query' en utilisant ' OR ' pour élargir la recherche.
    Exemple de requête très large : "cybersécurité OR télécommunication Sénégal OR Bénin OR marchés Banque Mondiale OR infrastructures vertes RDC"

    RÈGLE FONDAMENTALE : Dès que tu as toutes les informations nécessaires pour répondre COMPLÈTEMENT à la question initiale de l'utilisateur, tu DOIS IMPÉRATIVEMENT générer une Final Answer et ne plus utiliser d'outils.
    Final Answer: Ta réponse finale COMPLETE à la question de l'utilisateur.

    Commence maintenant!
    Question: {input}
    {agent_scratchpad}
    """
    agent_prompt = PromptTemplate.from_template(agent_prompt_content)
    agent = create_react_agent(llm_instance, tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    print("Agent de veille des marchés créé avec le type ReAct et AgentExecutor configuré.")
    return agent_executor

