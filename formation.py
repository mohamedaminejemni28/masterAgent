from langchain.prompts import PromptTemplate

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

import pandas as pd
from langchain.document_loaders import CSVLoader

file_path = "ALL8DATA.csv"
def load_formation(file_path, encoding="utf-8"):
    df = pd.read_csv(file_path, encoding=encoding)
    loader = CSVLoader(file_path=file_path, encoding=encoding)
    docs = loader.load()

    return df, docs

# Exemple d'utilisation

df, docs2 = load_formation(file_path)