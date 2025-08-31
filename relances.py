
# excel_handler.py
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np
# Exemple d’utilisation :
import os 
import sys
import os
import pandas as pd
import datetime

# Le chemin vers le fichier Excel
EXCEL_FILE_PATH = 'Agent_Commercial_IA_SFM_Gigantic_Dataset.xlsx'
PROSPECTS_SHEET = 'Relance_Clients' 
MARKET_AVIS_SHEET = 'Avis_Marches'

# Noms de colonnes utilisés dans la feuille Excel (basé sur votre capture d'écran)
CLIENT_ID_COLUMN = 'ID_Relance'
CLIENT_NAME_COLUMN = 'Client'
STATUS_COLUMN = 'Statut'
ACTION_COLUMN = 'Prochaine_Action'
DATE_COLUMN = 'Date_Dernière_Relance'

def _load_prospects_data() -> pd.DataFrame:
    """Charge une feuille spécifique d'un fichier Excel en DataFrame."""
    try:
        if not os.path.exists(EXCEL_FILE_PATH):
            print(f"Erreur: Le fichier {EXCEL_FILE_PATH} est introuvable.")
            return pd.DataFrame()

        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=PROSPECTS_SHEET)
        df.columns = df.columns.str.strip()
        
        # S'assurer que les colonnes nécessaires existent
        required_cols = [CLIENT_ID_COLUMN, CLIENT_NAME_COLUMN, STATUS_COLUMN, ACTION_COLUMN, DATE_COLUMN]
        if not all(col in df.columns for col in required_cols):
            print(f"Erreur: La feuille '{PROSPECTS_SHEET}' ne contient pas toutes les colonnes requises: {required_cols}.")
            return pd.DataFrame()

        # Convertir les colonnes de date et d'ID au bon format
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
        df[CLIENT_ID_COLUMN] = df[CLIENT_ID_COLUMN].astype(str)
        df[ACTION_COLUMN] = df[ACTION_COLUMN].astype(str)
        df[STATUS_COLUMN] = df[STATUS_COLUMN].astype(str)
        
        return df
    except (FileNotFoundError, ValueError) as e:
        print(f"Erreur lors de la lecture du fichier Excel: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Erreur inattendue lors de la lecture du fichier Excel: {e}")
        return pd.DataFrame()

from typing import List, Dict, Union

def read_prospects_for_action(action_type: str) -> Union[List[Dict], str]:
    """
    Reads prospects ready for a specific action, considering their status and last follow-up date.
    
    Args:
        action_type (str): The type of action ('all', 'email', 'whatsapp', etc.).

    Returns:
        Union[List[Dict], str]: List of prospects or a message if empty.
    """
    df = _load_prospects_data()
    if df.empty:
        return "Failed to read data. File not found or sheet is empty."
    
    today = datetime.now()
    
    df_filtered = df[
        (df['Statut'].str.lower().isin(['en attente', 'relancé', 'en négociation'])) &
        (df['Prochaine_Action'].str.lower() == action_type.lower()) &
        (df['Date_Dernière_Relance'].dt.date < (today - timedelta(days=30)).date())
    ].copy()

    if df_filtered.empty:
        return "No prospects found for this action."

    df_filtered = df_filtered.replace({np.nan: ''})

    if 'Email' not in df_filtered.columns:
        df_filtered['Email'] = df_filtered[CLIENT_NAME_COLUMN].apply(
            lambda client_name: f"{client_name.replace(' ', '').lower()}@example.com"
        )
    if 'Telephone' not in df_filtered.columns:
        df_filtered['Telephone'] = '123-456-7890'

    prospects_list = df_filtered[[
        'ID_Relance', 'Client', 'Email', 'Prochaine_Action', 'Statut', 'Date_Dernière_Relance', 'Telephone'
    ]].to_dict(orient='records')

    for prospect in prospects_list:
        if isinstance(prospect.get('Date_Dernière_Relance'), pd.Timestamp):
            prospect['Date_Dernière_Relance'] = prospect['Date_Dernière_Relance'].strftime('%Y-%m-%d')

    return prospects_list


# In[79]:


# agent_tools.py

from typing import List, Dict
from typing import List, Dict
from langchain.tools import tool  # Import the tool decorator

@tool
def get_prospects_for_relance(action_type: str) -> List[Dict]:
    """
    Récupère une liste des prospects qui nécessitent une relance pour un type d'action spécifique (e.g., 'email', 'whatsapp').
    L'action doit être une des suivantes: 'email', 'whatsapp', 'visio', 'appel'.
    Retourne une liste de dictionnaires avec les détails de chaque prospect.
    """
    return read_prospects_for_action(action_type)

@tool
def send_relance_email(email_address: str, subject: str, body: str) -> str:
    """
    Envoie un email de relance à un prospect.
    Prend en entrée l'adresse email du destinataire, le sujet de l'email, et le corps de l'email.
    Retourne une confirmation de l'envoi de l'email.
    """
    print("\n--- Simulation d'envoi d'email ---")
    print(f"À : {email_address}")
    print(f"Sujet : {subject}")
    print(f"Corps :\n{body}")
    print("-----------------------------------")
    return f"E-mail de relance simulé envoyé à {email_address}."

@tool
def send_whatsapp_message(telephone_number: str, message_body: str) -> str:
    """
    Envoie un message WhatsApp à un prospect.
    Prend en entrée le numéro de téléphone du destinataire et le corps du message.
    Retourne une confirmation de l'envoi du message WhatsApp.
    """
    print("\n--- Simulation d'envoi de message WhatsApp ---")
    print(f"À : {telephone_number}")
    print(f"Message : {message_body}")
    print("--------------------------------------------")
    return f"Message WhatsApp simulé envoyé au {telephone_number}."

@tool
def make_phone_call(telephone_number: str, reason: str) -> str:
    """
    Effectue un appel téléphonique à un prospect.
    Prend en entrée le numéro de téléphone et la raison de l'appel.
    Retourne une confirmation de l'appel téléphonique.
    """
    print("\n--- Simulation d'appel téléphonique ---")
    print(f"Numéro : {telephone_number}")
    print(f"Raison : {reason}")
    print("--------------------------------------")
    return f"Appel téléphonique simulé au {telephone_number} pour la raison : {reason}."

@tool
def schedule_visio_call(email_address: str, subject: str) -> str:
    """
    Planifie une visioconférence avec un prospect.
    Prend en entrée l'adresse e-mail du destinataire et le sujet de la réunion.
    Retourne une confirmation de la planification de la visioconférence.
    """
    print("\n--- Simulation de planification de visio ---")
    print(f"Destinataire : {email_address}")
    print(f"Sujet : {subject}")
    print("------------------------------------------")
    return f"Visioconférence simulée planifiée avec {email_address} pour le sujet : {subject}."

@tool
def mark_prospect_as_done(prospect_id: str, raison: str) -> str:
    """
    Met à jour le statut d'un prospect dans le fichier Excel.
    Prend en entrée l'ID du prospect à mettre à jour et la raison du changement de statut.
    Retourne une confirmation de la mise à jour.
    """
    print(f"Simulation de mise à jour: Prospect {prospect_id} - Statut changé en 'Relancé'.")
    success = update_prospect_status(prospect_id, 'Relancé')
    if success:
        return f"Statut du prospect {prospect_id} mis à jour avec la raison '{raison}'."
    else:
        return f"Erreur lors de la mise à jour du statut du prospect {prospect_id}."


# In[80]:


# agent_app_opensource.py
import pandas as pd
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Type
import os

# --- Configuration et variables globales ---
FILE_PATH = "Agent_Commercial_IA_SFM_Gigantic_Dataset.xlsx"
df_prospects = pd.DataFrame() # DataFrame global

# --- Définition des schémas d'entrée pour les outils ---

class RelanceEmailInput(BaseModel):
    email_address: str = Field(..., description="Adresse e-mail du destinataire.")
    subject: str = Field(..., description="Sujet de l'e-mail.")
    body: str = Field(..., description="Corps de l'e-mail.")

class RelanceWhatsappInput(BaseModel):
    telephone_number: str = Field(..., description="Numéro de téléphone du destinataire, incluant l'indicatif du pays.")
    message: str = Field(..., description="Message à envoyer.")

class RelanceAppelInput(BaseModel):
    telephone_number: str = Field(..., description="Numéro de téléphone à appeler.")
    reason: str = Field(..., description="Raison de l'appel.")

class RelanceVisioInput(BaseModel):
    email_address: str = Field(..., description="Adresse e-mail du destinataire.")
    subject: str = Field(..., description="Sujet de l'invitation de visioconférence.")

class UpdateProspectStatusInput(BaseModel):
    prospect_id: str = Field(..., description="ID du prospect à mettre à jour.")
    action: str = Field(..., description="Description de l'action effectuée (ex: 'Email envoyé', 'Appel effectué').")

# --- Fonctions de simulation d'actions ---

def send_email_simulation(email_address: str, subject: str, body: str) -> str:
    """Simule l'envoi d'un e-mail."""
    print(f"\n--- Simulation d'envoi d'email ---")
    print(f"À : {email_address}")
    print(f"Sujet : {subject}")
    print(f"Corps :\n{body}")
    print("-----------------------------------")
    return f"E-mail de relance simulé envoyé à {email_address}."

def send_whatsapp_simulation(telephone_number: str, message: str) -> str:
    """Simule l'envoi d'un message WhatsApp."""
    return f"Message WhatsApp simulé envoyé à {telephone_number} avec le message : '{message}'."

def make_phone_call_simulation(telephone_number: str, reason: str) -> str:
    """Simule un appel téléphonique."""
    print(f"\n--- Simulation d'appel téléphonique ---")
    print(f"Numéro : {telephone_number}")
    print(f"Raison : {reason}")
    print("--------------------------------------")
    return f"Appel téléphonique simulé au {telephone_number} pour la raison : {reason}."

def schedule_visio_call_simulation(email_address: str, subject: str) -> str:
    """Simule la planification d'une visioconférence."""
    print(f"\n--- Simulation de planification de visio ---")
    print(f"Destinataire : {email_address}")
    print(f"Sujet : {subject}")
    print("------------------------------------------")
    return f"Visioconférence simulée planifiée avec {email_address} pour le sujet : {subject}."

def update_prospect_status(prospect_id: str, action: str) -> str:
    """
    Simule la mise à jour du statut d'un prospect.
    Le nom de l'argument est 'action' pour être cohérent avec l'orchestrateur.
    """
    print(f"Simulation de mise à jour: Prospect {prospect_id} - Statut changé en '{action}'.")
    return f"Statut du prospect {prospect_id} mis à jour avec l'action : {action}."


# --- Définition des outils personnalisés (BaseTool) ---

class SendRelanceEmailTool(BaseTool):
    name: str = "send_relance_email"
    description: str = "Envoie un e-mail de relance à un prospect."
    # Correction de l'annotation de type ici
    args_schema: Type[BaseModel] = RelanceEmailInput

    def _run(self, email_address: str, subject: str, body: str) -> str:
        return send_email_simulation(email_address, subject, body)

class SendWhatsappMessageTool(BaseTool):
    name: str = "send_whatsapp_message"
    description: str = "Envoie un message WhatsApp à un prospect."
    # Correction de l'annotation de type ici
    args_schema: Type[BaseModel] = RelanceWhatsappInput

    def _run(self, telephone_number: str, message: str) -> str:
        return send_whatsapp_simulation(telephone_number, message)

class MakePhoneCallTool(BaseTool):
    name: str = "make_phone_call"
    description: str = "Passe un appel téléphonique à un prospect."
    # Correction de l'annotation de type ici
    args_schema: Type[BaseModel] = RelanceAppelInput

    def _run(self, telephone_number: str, reason: str) -> str:
        return make_phone_call_simulation(telephone_number, reason)

class ScheduleVisioCallTool(BaseTool):
    name: str = "schedule_visio_call"
    description: str = "Planifie une visioconférence avec un prospect."
    # Correction de l'annotation de type ici
    args_schema: Type[BaseModel] = RelanceVisioInput

    def _run(self, email_address: str, subject: str) -> str:
        return schedule_visio_call_simulation(email_address, subject)

class MarkProspectAsDoneTool(BaseTool):
    name: str = "mark_prospect_as_done"
    description: str = "Marque un prospect comme 'traité' après une relance réussie."
    # Correction de l'annotation de type ici
    args_schema: Type[BaseModel] = UpdateProspectStatusInput

    def _run(self, prospect_id: str, action: str) -> str:
        return update_prospect_status(prospect_id, action)

# --- Initialisation des objets Tool ---
send_relance_email = SendRelanceEmailTool()
send_whatsapp_message = SendWhatsappMessageTool()
make_phone_call = MakePhoneCallTool()
schedule_visio_call = ScheduleVisioCallTool()
mark_prospect_as_done = MarkProspectAsDoneTool()

# --- Fonction d'initialisation de l'agent de relance ---
def initialize_relance_agent(llm_instance):
    print("Agent de relance initialisé.")
    return [
        send_relance_email,
        send_whatsapp_message,
        make_phone_call,
        schedule_visio_call,
        mark_prospect_as_done
    ]