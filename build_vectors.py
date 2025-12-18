# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 14:37:20 2025

"""

import pandas as pd
import numpy as np
from mistralai import Mistral
import faiss
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

import json
from datetime import date
import dateutil.parser
import re
import pickle
import requests
from math import ceil
import time
import os

def json_parse(x):
    """
    Tente de parser une chaîne JSON en objet Python.

    Si l'entrée est une chaîne commençant et finissant par {}/[], 
    la fonction tente de la convertir avec json.loads. Si le parsing échoue,
    ou si l'entrée n'est pas une chaîne, la valeur originale est renvoyée.

    Args:
        x (any): Valeur à parser.

    Returns:
        any: Objet Python parsé ou valeur originale si échec.
    """
    
    if isinstance(x, str):
        x_strip = x.strip()
        # on teste si ça ressemble à un JSON encodé
        if (x_strip.startswith('[') and x_strip.endswith(']')) or (x_strip.startswith('{') and x_strip.endswith('}')):
            try:
                return json.loads(x_strip)
            except Exception:
                return x  # si parsing échoue, on garde la valeur originale
    return x

def convert_schedule_to_datetimes(raw_schedule):
    """
    Prend une liste de créneaux horaires de type :
    [
        {'begin': '2023-09-16T14:00:00+02:00', 'end': '2023-09-16T14:30:00+02:00'},
        ...
    ]
    
    Et retourne :
    [
        {'begin': datetime(...), 'end': datetime(...)},
        ...
    ]
    """
    if not raw_schedule:
        return np.nan

    converted = []
    for slot in raw_schedule:
        converted.append({
            "begin": dateutil.parser.isoparse(slot["begin"]),
            "end": dateutil.parser.isoparse(slot["end"])
        })

    return converted

def clean_html(text):
    """
    Nettoie un texte HTML en supprimant les balises et en normalisant les espaces.

    Remplace les listes HTML (<ul>, <ol>) par " : " puis remplace toutes les autres
    balises par esapce et enfin réduit les espaces multiples à un seul.

    Args:
        text (str): Texte HTML à nettoyer.

    Returns:
        str: Texte propre, sans balises et sans espaces superflus.
    """
    
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r"<\s*(ol|ul)(\s+[^>]*)?>", " : ", text, flags=re.IGNORECASE)
    
    text = re.sub(r"<[^>]+>", " ", text)
    
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def format_schedule_for_text(raw_schedule):
    """
    Transforme une liste de slots horaires OpenAgenda
    en un texte lisible pour un chunk vectorisé.
    
    Exemple d'entrée :
    [
        {'begin': datetime(...), 'end': datetime(...)},
        ...
    ]
    
    Sortie :
    Horaires :
    16/09/2023 14:00 - 14:30
    16/09/2023 14:30 - 15:00
    ...
    """
    if not raw_schedule:
        return ""

    lines = []
    for slot in raw_schedule:
        start = slot['begin']
        end   = slot['end']
        if not start:
            start_str = ''
        else:
            start_str = start.strftime("%d/%m/%Y %H:%M")
        if not end:
            end_str = ''
        else:
            end_str   = end.strftime("%H:%M")

        lines.append(f"{start_str} - {end_str}")

    return "\n".join(lines)
    
def build_text(row):
    """
    Construit un texte structuré à partir des champs d'une ligne de données.

    Les champs textuels sont nettoyés, formatés et concaténés afin de produire
    un texte unique destiné à l'indexation ou à la génération dans un pipeline RAG.

    Args:
        row (pd.Series): Ligne contenant les informations textuelles d'un évènement.

    Returns:
        str: Texte final concaténé et formaté.
    """
    
    r = row.fillna('')
    
    parts = [
        f"Titre : {r['title_fr']}",
        f"Description : {clean_html(r['description_fr'])}",
        f"Description longue : {clean_html(r['longdescription_fr'])}",
        f"Conditions : {r['conditions_fr']}",
        f"Mots clés : {r['keywords_fr']}",
        f"Horaires :\n{format_schedule_for_text(r['timings'])}",
        f"Lieu : {r['location_name']}",
        f"Lieu description : {r['location_description_fr']}",
        f"Adresse : {r['location_address']}",
        f"Téléphone : {r['location_phone']}",
        f"Site web : {r['location_website']}",
        f"Liens : {r['location_links']}",
        f"Accès / Itinéraire : {r['location_access_fr']}",
        f"Ville : {r['location_city']}",
        f"Département : {r['location_department']}",
        f"Type d'évènement : {r['attendancemode']}",
        f"Lien accès en ligne : {r['onlineaccesslink']}",
        f"Age minimum : {r['age_min']}",
        f"Age maximum : {r['age_max']}"
    ]
    return "\n".join([p for p in parts if p and str(p).strip() != ""])

def embed_batch(texts, client, batch_size=32, sleep_time=1.0, max_retries=10):
    """
    Génère des embeddings pour une liste de textes par lots avec gestion des retries.

    Les textes sont envoyés par batch à un client d'embeddings. En cas d'erreur,
    la fonction applique un backoff exponentiel et réessaie jusqu'à
    `max_retries` tentatives avant d'échouer.

    Args:
        texts (list of str): Textes à vectoriser.
        client: Client d'API d'embeddings.
        batch_size (int): Taille des lots d'embedding.
        sleep_time (float): Pause entre chaque batch.
        max_retries (int): Nombre maximum de tentatives par batch.

    Returns:
        list: Liste des vecteurs d'embeddings.
    """
    
    vectors = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        
        for attempt in range(1, max_retries + 1):
            try:
                # Appel embeddings Mistral pour tout le batch
                response = client.embeddings.create(
                    model="mistral-embed",
                    inputs=batch
                )
                
                # Chaque résultat correspond au texte de la même position
                vectors.extend([d.embedding for d in response.data])
                break  # sortie de la boucle retry si succès
            
            except Exception as e:
                wait_time = 2 ** (attempt - 1)  # backoff exponentiel : 1, 2, 4, 8, ...
                print(f"Erreur batch {i}-{i+len(batch)-1}, attempt {attempt}/{max_retries}: {e}")
                if attempt == max_retries:
                    raise RuntimeError(f"Echec après {max_retries} tentatives pour le batch {i}-{i+len(batch)-1}")
                print(f"Retry après {wait_time}s ...")
                time.sleep(wait_time)
        
        # Pause entre chaque batch pour limiter la charge
        time.sleep(sleep_time)
    
    return vectors

def build_faiss_indexes(df, cols_to_embed, client):
    """
    Construit des index FAISS pour plusieurs colonnes textuelles d'un DataFrame.

    Chaque colonne est vectorisée via des embeddings, puis indexée dans FAISS
    avec les identifiants métier présents dans la colonne 'uid'.

    Args:
        df (pd.DataFrame): DataFrame contenant les textes et la colonne 'uid'.
        cols_to_embed (list of str): Colonnes à vectoriser et indexer.
        client: Client utilisé pour la génération des embeddings.

    Returns:
        dict: Dictionnaire {nom_colonne: index FAISS}.
    """
    
    faiss_indexes = {}        # nom_colonne → index FAISS

    for col in cols_to_embed:
        print(f"--- Création FAISS pour colonne: {col} ---")

        # Récupération des textes et des IDs
        texts = df[col].fillna("").astype(str).tolist()
        ids = df["uid"].astype("int64").tolist()

        # Embedding
        vectors = embed_batch(texts, client)
        vectors = np.array(vectors).astype("float32")

        dim = vectors.shape[1]

        # Création d’un index avec support des IDs
        base_index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(base_index)

        # Ajout vecteurs + IDs métier
        index.add_with_ids(vectors, np.array(ids))

        # Sauvegarde en mémoire
        faiss_indexes[col] = index

    return faiss_indexes

def create_and_save_metadata(df, cols_meta, output_path="data/metadata.pkl"):
    """
    Crée un dict metadata par ligne et sauvegarde en pickle.
    
    Args:
        df (pd.DataFrame): dataframe contenant les colonnes metadata
        cols_meta (list of str): colonnes à inclure dans la metadata
        output_path (str): chemin du fichier pickle à créer
    
    Returns:
        metadata_list (list of dict): liste de metadata par ligne
    """
    
    # Créer la liste de dictionnaires
    metadata_list = df[cols_meta].fillna("").to_dict(orient="records")
    
    # Sauvegarder en pickle
    with open(output_path, "wb") as f:
        pickle.dump(metadata_list, f)
    
    print(f"▶ Metadata sauvegardée dans {output_path} ({len(metadata_list)} lignes)")
    
    return metadata_list

def save_indexes(faiss_indexes, prefix="data/indexes"):
    """
    Sauvegarde des index FAISS sur le disque.

    Chaque index est enregistré dans un fichier nommé selon sa clé,
    dans le dossier spécifié par `prefix`.

    Args:
        faiss_indexes (dict): Dictionnaire {nom: index FAISS}.
        prefix (str): Dossier de sauvegarde des index.
    """
    
    os.makedirs(prefix, exist_ok=True)

    for col, index in faiss_indexes.items():
        faiss.write_index(index, f"{prefix}/{col}.faiss")

    print("▶ Indexes FAISS sauvegardés ✔")

def concat_descriptions(row):
    """
    Concatène et nettoie les champs 'description_fr' et 'longdescription_fr'.

    La fonction combine les deux descriptions lorsqu'elles sont présentes,
    les nettoie avec `clean_html` et retourne le texte résultant.
    Si aucun champ valide n'est disponible, une chaîne vide est renvoyée.

    Args:
        row (dict or pandas.Series): Ligne contenant les descriptions en français.

    Returns:
        str: Description nettoyée et concaténée.
    """
    
    if isinstance(row['description_fr'], str):
        if isinstance(row['longdescription_fr'], str):
            return clean_html("\n".join([row['description_fr'],row['longdescription_fr']]))
        else:
            return clean_html(row['description_fr'])
    else:
        if isinstance(row['longdescription_fr'], str):
            return clean_html(row['longdescription_fr'])
        else:
            return ''

def get_date_range(years_back=1, years_forward=1):
    """Renvoie les dates min et max pour la requête OpenDataSoft au format YYYY-MM-DD."""
    today = date.today()
    start = (today - relativedelta(years=years_back)).strftime("%Y-%m-%d")
    end = (today + relativedelta(years=years_forward)).strftime("%Y-%m-%d")
    return start, end

def fetch_opendatasoft_events(region="Auvergne-Rhône-Alpes", start_date=None, end_date=None):
    """
    Récupère tous les événements OpenDataSoft pour une région et une période données.
    Gère le paging si plus de 100 résultats.
    """
    headers = {"Content-Type": "application/json"}
    base_url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records/"
    
    params = f"?limit=100&offset=0&where=lastdate_end+%3E%3D+%22{start_date}%22+AND+firstdate_begin+%3C%3D+%22{end_date}%22+AND+location_region+%3D+%22{region}%22"
    response = requests.get(base_url + params, headers=headers).json()
    
    total_count = response['total_count']
    results = pd.DataFrame(data=response['results']).dropna(how='all')
    
    if total_count > 100:
        additional_requests = ceil((total_count - 100) / 100)
        for i in range(additional_requests):
            offset = 100 + 100 * i
            params_offset = f"?limit=100&offset={offset}&where=lastdate_end+%3E%3D+%22{start_date}%22+AND+firstdate_begin+%3C%3D+%22{end_date}%22+AND+location_region+%3D+%22{region}%22"
            r = requests.get(base_url + params_offset, headers=headers).json()
            df_new = pd.DataFrame(r['results']).dropna(how='all').dropna(axis=1, how='all')
            if not df_new.empty:
                results = pd.concat([results, df_new], axis=0)
    return results

def preprocess_events(df):
    """Nettoie et transforme le DataFrame OpenDataSoft pour l'indexation et l'embedding."""
    df = df.map(json_parse)
    df['descriptions'] = df.apply(concat_descriptions, axis=1)
    
    cols_text = ['descriptions']
    cols_meta = ['uid','location_address','location_phone','location_website','location_links','location_tags',
                 'location_access_fr','location_city','location_department','attendancemode','onlineaccesslink',
                 'age_min','age_max','timings','title_fr','conditions_fr','keywords_fr','location_name','location_description_fr']
    
    df_lite = df[['description_fr','longdescription_fr'] + cols_text + cols_meta].copy()
    
    # transformations mineures
    df_lite['timings'] = df_lite['timings'].apply(convert_schedule_to_datetimes)
    df_lite['attendancemode'] = df_lite['attendancemode'].apply(lambda x: x['label']['fr'])
    
    # création du texte combiné pour LLM
    df_lite['text'] = df_lite.apply(build_text, axis=1)
    cols_meta.append('text')
    
    return df_lite, cols_text, cols_meta

def main():
    """Pipeline principal : récupération, nettoyage, embedding, indexation et sauvegarde."""
    print("Chargement data depuis Opendatasoft ...")
    start_date, end_date = get_date_range()
    events_df = fetch_opendatasoft_events(start_date=start_date, end_date=end_date)
    print("Chargement terminé.")
    
    print("Cleanup et transformation en cours ...")
    df_lite, cols_text, cols_meta = preprocess_events(events_df)
    print("Cleanup et transformation terminés")
    
    api_key = 'U0q6FKSPa9QLqebhTh4XMG7t02N72k86'
    client = Mistral(api_key=api_key)
    
    faiss_indexes = build_faiss_indexes(df_lite, cols_text, client)
    save_indexes(faiss_indexes)
    
    metadata = create_and_save_metadata(df_lite, cols_meta)
    
if __name__ == "__main__":
    main()