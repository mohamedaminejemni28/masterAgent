from langchain.prompts import PromptTemplate
import pandas as pd
import time
import random
from urllib.parse import urlparse
import os
from ddgs import DDGS  
import re
from langchain.document_loaders import CSVLoader


def prompt_concurrent():
    prompt_concurrent = PromptTemplate.from_template("""
    Tu es un assistant spécialisé dans l'analyse de concurrents à partir de documents structurés.

    Chaque document contient les champs suivants :
    - Produit
    - Titre
    - URL
    - Nom
    - Description

    Instructions :
    - Fournis la réponse uniquement sous forme de liste claire contenant : **Produit, Titre, URL, Nom, Description**.
    - Si plusieurs documents sont fournis, liste-les tous.
    - Si aucun document pertinent n’est trouvé, réponds : *"Aucun concurrent trouvé correspondant à votre demande."*
    - N’ajoute rien d’autre en dehors de la liste.

    Documents disponibles :
    {context}

    Question de l’utilisateur : {question}
    """)
    return prompt_concurrent

COMPETITORS_EXCEL_FILE = 'Agent_Commercial_IA_SFM_Gigantic_Dataset.xlsx'
COMPETITORS_SHEET_NAME = 'Veille_Concurrents'
OUTPUT_CSV_FILE = 'concurrents_sfm.csv'
PRODUCTS = ['Formation IA', 'Consulting', 'Cybersécurité', 'Audit SI', 'Cloud']
MAX_RESULTS = 200 
TARGET_COMPETITORS = 100

EXCLUDED_DOMAINS = [
    'wikipedia.org', 'reddit.com', 'facebook.com', 'google.com', 'youtube.com',
    'twitter.com', 'linkedin.com', 'instagram.com', 'tiktok.com', 'pinterest.com',
    'amazon.com', 'healthline.com', 'webmd.com', 'apple.com', 'microsoft.com',
    'quora.com', 'scribd.com', 'indeed.com', 'marketwatch.com', 'bloomberg.com',
    'nasdaq.com', 'yahoo.com', 'answers.com', 'envato.com', 'brainyquote.com',
    'forbes.com', 'crn.com', 'cloudtango.net', 'trainingindustry.com', 'inven.ai',
    'clutch.co', 'sortlist.com', 'expertise.com', 'mordorintelligence.com',
    'cisa.gov', 'comptia.org', 'geeksforgeeks.org', 'techtarget.com',
    'purdueglobal.edu', 'caltech.edu', 'zhihu.com', 'theiia.org', 'shine.com',
    'statista.com', 'accaglobal.com', 'builtin.com', 'managementconsulted.com',
    'designrush.com', 'cbinsights.com', 'emerj.com', 'grandviewresearch.com'
]

VALID_KEYWORDS = {
    'Formation IA': [  'bootcamp',  'machine learning', 'deep learning', 'AI entreprise', 'artificial intelligence '],
    'Consulting': ['consulting', 'consultancy', 'advisory', 'management', 'strategy', 'digital transformation', 'IT consulting', 'business consulting', 'technology consulting'],
    'Cybersécurité': ['cybersecurity', 'security', 'protection', 'threat', 'secure', 'network security', 'cyber defense', 'information security', 'cyber protection'],
    'Audit SI': ['audit', 'IT audit', 'compliance', 'risk management', 'information systems', 'security audit', 'IT compliance', 'system audit'],
    'Cloud': [    "cloud", "cloud services", "cloud computing", "SaaS", "PaaS", "IaaS",
    "cloud infrastructure", "cloud solutions", "cloud provider",
    "enterprise cloud", "cloud migration", "cloud security", 
    "hybrid cloud", "private cloud", "public cloud", "multi-cloud",
    "cloud strategy", "cloud management", "cloud optimization"]
}

def normalize_domain(domain):
    if not domain:
        return ""
    domain = re.sub(r'^(www\d?\.|blog\.|news\.)', '', domain.lower())
    return domain

def is_valid_domain(domain):
    if not domain:
        return False
    normalized_domain = normalize_domain(domain)
    if not any(normalized_domain.endswith(ext) for ext in ['.com', '.co', '.io', '.org', '.net', '.ai', '.tech', '.eu', '.fr', '.de']):
        return False
    return not any(excluded in normalized_domain for excluded in EXCLUDED_DOMAINS)

def is_valid_content(title, description, produit):
    if not title and not description:
        return False
    keywords = VALID_KEYWORDS.get(produit, [])
    content = (title.lower() + " " + description.lower()).strip()
    return any(keyword in content for keyword in keywords)

def search_duckduckgo(produit, max_results=MAX_RESULTS):
    competitors = []
    query_map = {
        'Formation IA': [
            'artificial intelligence training company',
            'machine learning course provider',
            'AI education entreprise',
            'deep learning training firm',
            'AI certification company',
            'formation intelligence artificielle entreprise'
        ],
        'Consulting': [
            'IT consulting firm',
            'digital transformation consultancy',
            'management consulting company',
            'technology advisory firm',
            'business consulting services',
            'conseil en technologie entreprise'
        ],
        'Cybersécurité': [
            'cybersecurity solutions provider',
            'network security company',
            'cyber threat protection firm',
            'information security services',
            'cybersecurity consulting',
            'sécurité informatique entreprise'
        ],
        'Audit SI': [
            'IT audit services',
            'information systems audit firm',
            'compliance audit company',
            'security audit provider',
            'IT risk management firm',
            'audit informatique entreprise'
        ],
        'Cloud': [
            'cloud services provider',
            'cloud computing company',
            'SaaS company',
            'cloud infrastructure firm',
            'cloud solutions company',
            'services cloud entreprise'
        ]
    }
    queries = query_map.get(produit, [f"{produit} company"])
    
    for query in queries:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results // len(queries)))
            if not results:
                print(f"Aucun résultat DuckDuckGo pour la requête '{query}'")
                continue
            for result in results:
                domain = urlparse(result.get("href", "")).netloc
                title = result.get("title", "")
                description = result.get("body", "")
                if is_valid_domain(domain) and is_valid_content(title, description, produit):
                    competitors.append({
                        "Produit": produit,
                        "Titre": title,
                        "URL": result.get("href", ""),
                        "nom": normalize_domain(domain),
                        "Description": description
                    })
            print(f"{len(competitors)} résultats valides trouvés pour {produit} avec la requête '{query}'")
        except Exception as e:
            print(f"Erreur DuckDuckGo pour la requête '{query}': {e}")
        time.sleep(random.uniform(5, 10))  #
    return competitors

def load_excel_competitors(file_path, sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        competitors = []
        for _, row in df.iterrows():
            domain = urlparse(row.get("Site_Web", "")).netloc if row.get("Site_Web") else ""
            title = row.get("Entreprise", "")
            description = row.get("Description", "")
            produit = row.get("Produit", "")
            if is_valid_domain(domain) and is_valid_content(title, description, produit):
                competitors.append({
                    "Produit": produit,
                    "Titre": title,
                    "URL": row.get("Site_Web", ""),
                    "nom": normalize_domain(domain),
                    "Description": description
                })
        print(f"{len(competitors)} concurrents valides chargés depuis le fichier Excel")
        return competitors
    except Exception as e:
        print(f"Erreur lors du chargement du fichier Excel: {e}")
        return []

# Fonction principale pour collecter les concurrents
def collect_competitors():
    all_competitors = []

    # 1. Chargement depuis le fichier Excel
    if os.path.exists(COMPETITORS_EXCEL_FILE):
        print("Chargement des concurrents depuis le fichier Excel...")
        excel_competitors = load_excel_competitors(COMPETITORS_EXCEL_FILE, COMPETITORS_SHEET_NAME)
        all_competitors.extend(excel_competitors)
    else:
        print(f"Fichier Excel '{COMPETITORS_EXCEL_FILE}' non trouvé. Utilisation de DuckDuckGo.")

    # 2. Recherche via DuckDuckGo
    df_temp = pd.DataFrame(all_competitors).drop_duplicates(subset=["nom"], keep="first")
    if len(df_temp) < TARGET_COMPETITORS:
        print(f"Nombre de concurrents uniques ({len(df_temp)}) insuffisant, recherche via DuckDuckGo...")
        for produit in PRODUCTS:
            print(f"Recherche des concurrents pour : {produit}")
            competitors = search_duckduckgo(produit)
            all_competitors.extend(competitors)

    # Création d'un DataFrame et suppression des doublons
    df = pd.DataFrame(all_competitors)
    if df.empty:
        print("Aucun concurrent trouvé. Vérifiez le fichier Excel ou les requêtes DuckDuckGo.")
        return df

    df_unique = df.drop_duplicates(subset=["nom"], keep="first")

    # Sauvegarde des résultats
    df_unique.to_csv(OUTPUT_CSV_FILE, index=False, encoding="utf-8")
    print(f"Résultats enregistrés dans '{OUTPUT_CSV_FILE}' ({len(df_unique)} concurrents uniques trouvés)")

    # Vérification de l'objectif
    if len(df_unique) < TARGET_COMPETITORS:
        print(f"Attention : seulement {len(df_unique)} concurrents uniques trouvés. Essayez d'exécuter localement, d'augmenter MAX_RESULTS, ou d'ajouter des sources comme Clutch.co.")
    else:
        print(f"Objectif atteint : {len(df_unique)} concurrents uniques trouvés.")



    return df_unique

loader = CSVLoader(file_path=OUTPUT_CSV_FILE, encoding="utf-8")
docs3 = loader.load()