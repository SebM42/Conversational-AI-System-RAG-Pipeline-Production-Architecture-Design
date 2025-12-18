# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 11:15:26 2025

"""
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from mistralai import Mistral
import faiss
import pickle
import numpy as np

from datetime import datetime, date
from fastapi import FastAPI
from pydantic import BaseModel

import json
import re
import time

system_template_extract_filters = """
Tu es un assistant qui reçoit une question utilisateur sur des événements.
Ta tâche :
1. Identifier les filtres applicables sur les metadata (format JSON)
2. Transformer le prompt en une description des thématiques recherchées, en français

Constraints:
- location_city doit être une ville française existante et être explicitement dans la question (corrige les accents si besoin), sinon null
- location_department doit être le nom du département et être explicitement dans la question (corrige les accents si besoin), sinon null
- attendancemode ne peut prendre que ['Sur place', 'En ligne', 'Mixte'], sinon null
- age_min est l'age minimum requis, pour rappel un adulte est une personne qui a 18 ans ou plus'
- "query" est le prompt transformé en texte précis décrivant la ou les thématiques recherchées, retire toute référence de temporalité, ainsi que le terme "évènement" et ses synonymes

Répond toujours en JSON comme ceci :

{{
  "filters": {{"location_city": null,"location_department": null, "age_min": null, "age_max": null, "attendancemode": null}},
  "query": ""
}}
"""

system_template_final_response = """
Tu es un assistant qui doit produire une réponse à partir d'une question utilisateur et d'une liste de données récupérées sur les évènements disponibles.
Pour chaque évènement de la liste, tu dois les lister un par un selon le modèle suivant :
1. Préciser le titre, le lieu, les modalités et le lien vers le site s'il existe'
2. Si plusieurs dates, en proposer jusqu'à 4, en précisant les horaires
3. Formuler une description de l'évènement sur quelques lignes

N'invente rien. Si des informations manquent, ne les mentionnent pas.

Si des évènements semblent ne pas être en accord avec la question, commence par lister les évènements qui correspondent, puis précise que les évènements
suivants ne correspondent pas mais sont les plus proches de la demande initiale, et liste les selon le même modèle.

Données récupérées : {text}
"""

def embed_text(text, client, sleep_time=1.0, max_retries=5):
    """
    Génère l'embedding d'un texte via l'API Mistral en gérant les erreurs temporaires.

    La fonction tente d'appeler le modèle d'embedding `mistral-embed`.
    En cas d'échec, elle applique une stratégie de retry avec backoff exponentiel.

    Args:
        text (str): Texte à transformer en embedding.
        client: Client Mistral configuré pour accéder à l'API.
        sleep_time (float, optional): Temps d'attente entre deux appels réussis (non utilisé ici).
        max_retries (int, optional): Nombre maximal de tentatives avant échec définitif.

    Returns:
        list[float]: Vecteur embedding correspondant au texte.

    Raises:
        RuntimeError: Si toutes les tentatives échouent.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = client.embeddings.create(
                model="mistral-embed",
                inputs=[text]  # note la liste
            )
            # récupérer le vecteur
            embedding = response.data[0].embedding
            return embedding
        
        except Exception as e:
            wait_time = 2 ** (attempt - 1)
            print(f"Erreur embed texte, attempt {attempt}/{max_retries}: {e}")
            if attempt == max_retries:
                raise RuntimeError(f"Echec après {max_retries} tentatives")
            print(f"Retry après {wait_time}s ...")
            time.sleep(wait_time)
            
class MistralEmbedWrapper(Embeddings):
    """
    Wrapper pour utiliser le client Mistral comme générateur d'embeddings
    compatible avec l'interface LangChain `Embeddings`.

    Args:
        client: Instance du client Mistral utilisé pour générer les embeddings.
        sleep_time (float, optional): Temps d'attente entre tentatives en secondes pour gérer la charge. Default=1.0.
        max_retries (int, optional): Nombre maximal de tentatives en cas d'erreur lors de l'appel du client. Default=5.

    Methods:
        embed_documents(texts): Retourne une liste de vecteurs embeddings pour une liste de textes.
        embed_query(text): Retourne le vecteur embedding pour une seule requête.
    """
    def __init__(self, client, sleep_time=1.0, max_retries=5):
        self.client = client
        self.sleep_time = sleep_time
        self.max_retries = max_retries
    
    def embed_documents(self, texts):
        return [embed_text(t, self.client) for t in texts]

    def embed_query(self, text):
        return embed_text(text, self.client)

def filter_metadata(metadata_list, filters):
    """
    Filtre une liste de métadonnées d'événements selon des critères simples
    et retourne les UIDs des documents correspondant.

    Args:
        metadata_list (list of dict): Liste de dictionnaires contenant les métadonnées
            de chaque événement. Chaque dictionnaire doit inclure au minimum les clés
            'uid', 'location_city', 'location_department', et 'attendancemode'.
        filters (dict): Dictionnaire des critères de filtrage avec les clés possibles :
            - 'location_city' (str ou None) : ne garder que les événements dans cette ville.
            - 'location_department' (str ou None) : ne garder que les événements dans ce département.
            - 'attendancemode' (str ou None) : ne garder que les événements correspondant à ce mode de participation.

    Returns:
        list: Liste des UIDs des événements qui correspondent à tous les filtres spécifiés.
    """
    
    matching_uids = []

    # convertir les dates filtrées en datetime pour comparer
    #filter_dates = [datetime.strptime(d, "%Y-%m-%d") for d in filters.get('timings', [])]

    for doc in metadata_list:
        # Vérification location_city
        if filters['location_city'] is not None:
            if doc.get('location_city') != filters['location_city']:
                continue

        # Vérification location_department
        if filters['location_department'] is not None:
            if doc.get('location_department') != filters['location_department']:
                continue

        # Vérification attendancemode
        if filters['attendancemode'] is not None:
            if doc.get('attendancemode') != filters['attendancemode']:
                continue
            
        # Vérification agemin
        if filters['age_min'] is not None:
            if doc.get('age_min') != filters['age_min']:
                continue
            
        # Vérification agemax
        if filters['age_max'] is not None:
            if doc.get('age_max') != filters['age_max']:
                continue

        # Vérification timings : au moins une date en commun
        '''doc_intervals = doc.get('timings', [])
        if filter_dates:
            match_found = False
            for f_date in filter_dates:
                for interval in doc_intervals:
                    if interval['begin'].date() <= f_date.date() <= interval['end'].date():
                        match_found = True
                        break
                if match_found:
                    break
            if not match_found:
                continue'''

        matching_uids.append(doc['uid'])

    return matching_uids

def load_client_and_vectorstore(api_key: str, index_path: str, metadata_path: str):
    """
    Initialise le client Mistral ainsi que le vectorstore FAISS à partir
    d'un index sauvegardé et d'un fichier de métadonnées.

    La fonction :
    1. Instancie le client Mistral.
    2. Charge l'index FAISS depuis le disque.
    3. Charge les métadonnées depuis un fichier pickle.
    4. Construit un docstore LangChain (InMemoryDocstore).
    5. Initialise un vectorstore FAISS prêt pour la recherche vectorielle.

    Args:
        api_key (str): Clé API Mistral.
        index_path (str): Chemin vers le fichier FAISS (.faiss).
        metadata_path (str): Chemin vers le fichier pickle contenant les métadonnées.

    Returns:
        tuple:
            - client: Instance du client Mistral.
            - vectorstore (FAISS): Vectorstore LangChain prêt à l'emploi.
            - metadata_list (list[dict]): Liste brute des métadonnées chargées.

    Raises:
        FileNotFoundError: Si l'index FAISS ou le fichier metadata est introuvable.
        pickle.UnpicklingError: Si le fichier metadata est invalide.
    """
    
    client = Mistral(api_key=api_key)
    faiss_index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata_list = pickle.load(f)

    metadata_store = {item["uid"]: item for item in metadata_list}

    docstore = InMemoryDocstore({
        uid: Document(page_content=item["text"], metadata=item)
        for uid, item in metadata_store.items()
    })

    index_to_docstore_id = {int(uid): str(uid) for uid in metadata_store.keys()}

    vectorstore = FAISS(
        index=faiss_index,
        embedding_function=MistralEmbedWrapper(client),
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    return client, vectorstore, metadata_list

def extract_filters_and_query(client, system_prompt_template, user_question: str):
    """
    Interroge le LLM afin d'extraire des filtres structurés et une requête
    de recherche à partir d'une question utilisateur en langage naturel.

    La fonction :
    1. Appelle le modèle de chat Mistral.
    2. Extrait un bloc JSON (optionnellement encadré par ```json).
    3. Parse le JSON pour récupérer les filtres et la requête finale.

    Args:
        client: Client LLM exposant `chat.complete`.
        system_prompt_template: Template de prompt système
        user_question (str): Question posée par l'utilisateur.

    Returns:
        tuple:
            - filters (dict): Dictionnaire de filtres métier extraits.
            - query (str): Requête textuelle nettoyée pour la recherche vectorielle.

    Raises:
        json.JSONDecodeError: Si la réponse du LLM n'est pas un JSON valide.
        KeyError: Si les clés attendues (`filters`, `query`) sont absentes.
    """
    
    response = client.chat.complete(
        model="magistral-small-2509",
        messages=[{
            "role": "system",
            "content": system_prompt_template.format()
        }, {
            "role": "user",
            "content": user_question
        }]
    )

    raw_text = response.choices[0].message.content[1].text
    pattern = r"```json\*?(.*?)```"
    match = re.search(pattern, raw_text, re.DOTALL)
    json_text = match.group(1).strip() if match else raw_text.strip()

    parsed_json = json.loads(json_text)
    return parsed_json["filters"], parsed_json["query"]

def score_and_filter_metadata(vectorstore, metadata_list, query_text, filters):
    """
    Effectue une recherche vectorielle sur les documents et applique des filtres métier
    sur les métadonnées résultantes.

    La fonction :
    1. Exécute une recherche de similarité vectorielle à partir du texte de requête.
    2. Associe chaque document à son score de similarité.
    3. Filtre les résultats selon des critères métier (localisation, type d'événement, etc.).

    Args:
        vectorstore: Instance de VectorStore (ex: FAISS) supportant la méthode
            `similarity_search_with_score`.
        metadata_list (list[dict]): Liste complète des métadonnées (utilisée pour définir `k`).
        query_text (str): Texte de requête utilisé pour la recherche vectorielle.
        filters (dict): Dictionnaire de filtres métier à appliquer sur les métadonnées.

    Returns:
        list[dict]: Liste des métadonnées filtrées enrichies du score de similarité.
    """
    
    docs_scores = vectorstore.similarity_search_with_score(query_text, k=len(metadata_list))

    metadata_with_score = []
    for doc, score in docs_scores:
        md = doc.metadata.copy()
        md["score"] = score
        metadata_with_score.append(md)

    filtered_uids = filter_metadata(metadata_with_score, filters)
    results = [d for d in metadata_with_score if d.get("uid") in filtered_uids]

    return results

def generate_final_response(client, system_prompt_template, user_question: str, text: str):
    """
    Génère la réponse finale du LLM à partir d'un contexte textuel filtré.

    La fonction appelle le modèle de génération Mistral en lui fournissant :
    - un prompt système contenant le texte de contexte,
    - la question utilisateur.

    Elle extrait ensuite le texte final de la réponse ainsi que
    la partie "thinking" (raisonnement interne du modèle).

    Args:
        client: Client Mistral configuré pour l'appel au modèle de génération.
        system_prompt_template: Template de prompt système (PromptTemplate ou équivalent)
            devant accepter la variable `text`.
        user_question (str): Question posée par l'utilisateur.
        text (str): Texte de contexte filtré (souvent issu d'un RAG).

    Returns:
        tuple[str, str]: 
            - final_raw_text : réponse finale générée par le LLM,
            - final_thinking : raisonnement interne du modèle.
    """
    
    response = client.chat.complete(
        model="magistral-small-2509",
        messages=[{
            "role": "system",
            "content": system_prompt_template.format(text=text)
        }, {
            "role": "user",
            "content": user_question
        }]
    )
    final_raw_text = response.choices[0].message.content[1].text
    final_thinking = response.choices[0].message.content[0].thinking[0].text
    
    return final_raw_text, final_thinking

def default_serializer(obj):
    """
    Sérialiseur personnalisé pour json.dumps.

    Gère les types non JSON-serializables courants :
    - datetime / date -> ISO 8601
    - numpy scalars (float32, int64, etc.) -> types Python natifs
    - numpy arrays -> listes Python

    Args:
        obj: Objet Python à sérialiser.

    Returns:
        Objet sérialisable JSON (str, int, float, list).

    Raises:
        TypeError: Si le type n'est pas supporté.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, np.generic):  # float32, int64, etc.
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(f"Type {type(obj)} non sérialisable")

## init
app = FastAPI(title="RAG Mistral API")

# Modèle de requête
class QuestionRequest(BaseModel):
    user_question: str

API_KEY = 'U0q6FKSPa9QLqebhTh4XMG7t02N72k86'
VECTOR_INDEX_PATH = "data/indexes/descriptions.faiss"
METADATA_PATH = "data/metadata.pkl"

client, vectorstore, metadata_list = load_client_and_vectorstore(
    api_key=API_KEY,
    index_path=VECTOR_INDEX_PATH,
    metadata_path=METADATA_PATH
)
print("Vector store loaded\n")

system_prompt_extract_filters = PromptTemplate(template=system_template_extract_filters)
system_prompt_final_response = PromptTemplate(
    input_variables=["text"],
    template=system_template_final_response
)

@app.post("/ask")
def ask_question(request: QuestionRequest):
    user_question = request.user_question
    print(f"Question : {user_question}\n")
    
    #try:
    # Extraction des filtres et de la requête
    filters, query_text = extract_filters_and_query(client, system_prompt_extract_filters, user_question)
    print(f"Filtres :\n{json.dumps(filters, indent=4, ensure_ascii=False)}\n")
    print(f"Thematique : {query_text}\n")

    # Recherche vectorielle et filtrage
    results = score_and_filter_metadata(vectorstore, metadata_list, query_text, filters)

    if not results:
        return {"final_raw_text": "", "message": "Aucun résultat ne correspond aux filtres."}

    # Extraire les textes des 3 premiers résultats
    top_results = results[:3]
    final_text = [r["text"] for r in top_results]

    # Générer la réponse finale
    final_raw_text, final_thinking = generate_final_response(
        client, system_prompt_final_response, user_question, final_text
    )
    print(f'Resultat 1 :\n{json.dumps(top_results[0], indent=4, ensure_ascii=False, default=default_serializer)}\n')
    if len(top_results) > 1:
        print(f'Resultat 2 :\n{json.dumps(top_results[1], indent=4, ensure_ascii=False, default=default_serializer)}\n')
    if len(top_results) > 2:
        print(f'Resultat 3 :\n{json.dumps(top_results[2], indent=4, ensure_ascii=False, default=default_serializer)}\n')
    
    print(f"Reflexion finale :\n {final_thinking}\n")
    print(f"Reponse finale :\n {final_raw_text}\n")

    return {"final_raw_text": final_raw_text}
