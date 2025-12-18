# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 10:14:26 2025

"""

from build_vectors import fetch_opendatasoft_events
from build_vectors import get_date_range
from build_vectors import concat_descriptions
from build_vectors import save_indexes
from build_vectors import create_and_save_metadata
from build_vectors import build_faiss_indexes
from build_vectors import embed_batch
from build_vectors import build_text
from build_vectors import format_schedule_for_text
from build_vectors import clean_html
from build_vectors import convert_schedule_to_datetimes
from build_vectors import json_parse

from unittest.mock import patch, MagicMock
from datetime import datetime, date
import pytest
import pandas as pd
import numpy as np
import pickle

### TESTS json_parse
def test_json_parse_valid_dict():
    s = '{"a":1,"b":2}'
    result = json_parse(s)
    assert isinstance(result, dict)
    assert result == {"a": 1, "b": 2}


def test_json_parse_valid_list():
    s = '[1, 2, 3]'
    result = json_parse(s)
    assert isinstance(result, list)
    assert result == [1, 2, 3]


def test_json_parse_invalid_json():
    s = '{"a":1,,}'
    result = json_parse(s)
    assert result == s  # garde la valeur originale


def test_json_parse_non_string():
    for val in [123, None, True, {"a":1}]:
        assert json_parse(val) == val


def test_json_parse_string_no_json():
    s = "Just a regular string"
    assert json_parse(s) == s

### TESTS convert_schedule_to_datetimes
def test_convert_schedule_normal_case():
    raw_schedule = [
        {"begin": "2023-09-16T14:00:00+02:00", "end": "2023-09-16T14:30:00+02:00"},
        {"begin": "2023-09-16T14:30:00+02:00", "end": "2023-09-16T15:00:00+02:00"},
    ]

    result = convert_schedule_to_datetimes(raw_schedule)

    assert isinstance(result, list)
    assert all(isinstance(slot["begin"], datetime) for slot in result)
    assert all(isinstance(slot["end"], datetime) for slot in result)
    assert result[0]["begin"].isoformat() == "2023-09-16T14:00:00+02:00"
    assert result[1]["end"].isoformat() == "2023-09-16T15:00:00+02:00"


def test_convert_schedule_empty_input():
    assert np.isnan(convert_schedule_to_datetimes([]))
    assert np.isnan(convert_schedule_to_datetimes(None))


def test_convert_schedule_single_slot():
    raw_schedule = [
        {"begin": "2023-01-01T10:00:00+00:00", "end": "2023-01-01T11:00:00+00:00"}
    ]
    result = convert_schedule_to_datetimes(raw_schedule)
    assert len(result) == 1
    assert result[0]["begin"].isoformat() == "2023-01-01T10:00:00+00:00"
    assert result[0]["end"].isoformat() == "2023-01-01T11:00:00+00:00"


def test_convert_schedule_invalid_format():
    raw_schedule = [
        {"begin": "invalid", "end": "2023-01-01T11:00:00+00:00"}
    ]
    with pytest.raises(ValueError):
        convert_schedule_to_datetimes(raw_schedule)

### TESTS clean_html
def test_clean_html_normal_text():
    html = "<p>Hello <b>world</b>!</p>"
    result = clean_html(html)
    assert result == "Hello world !"


def test_clean_html_with_lists():
    html = "<ul><li>Item1</li><li>Item2</li></ul>"
    result = clean_html(html)
    assert result == ": Item1 Item2"


def test_clean_html_multiple_spaces():
    html = "<p>Hello     world</p>"
    result = clean_html(html)
    assert result == "Hello world"


def test_clean_html_non_string_input():
    assert clean_html(None) == ""
    assert clean_html(123) == ""


def test_clean_html_empty_string():
    assert clean_html("") == ""

### TEST format_schedule_for_text
def test_format_schedule_for_text_normal_case():
    raw_schedule = [
        {
            "begin": datetime(2023, 9, 16, 14, 0),
            "end": datetime(2023, 9, 16, 14, 30),
        },
        {
            "begin": datetime(2023, 9, 16, 14, 30),
            "end": datetime(2023, 9, 16, 15, 0),
        },
    ]

    result = format_schedule_for_text(raw_schedule)

    expected = (
        "16/09/2023 14:00 - 14:30\n"
        "16/09/2023 14:30 - 15:00"
    )

    assert result == expected


def test_format_schedule_for_text_empty_input():
    assert format_schedule_for_text([]) == ""
    assert format_schedule_for_text(None) == ""


def test_format_schedule_for_text_missing_begin():
    raw_schedule = [
        {
            "begin": None,
            "end": datetime(2023, 9, 16, 14, 30),
        }
    ]

    result = format_schedule_for_text(raw_schedule)

    assert result == " - 14:30"


def test_format_schedule_for_text_missing_end():
    raw_schedule = [
        {
            "begin": datetime(2023, 9, 16, 14, 0),
            "end": None,
        }
    ]

    result = format_schedule_for_text(raw_schedule)

    assert result == "16/09/2023 14:00 - "


def test_format_schedule_for_text_multiple_lines_count():
    raw_schedule = [
        {
            "begin": datetime(2023, 1, 1, 10, 0),
            "end": datetime(2023, 1, 1, 11, 0),
        },
        {
            "begin": datetime(2023, 1, 2, 12, 0),
            "end": datetime(2023, 1, 2, 13, 0),
        },
        {
            "begin": datetime(2023, 1, 3, 14, 0),
            "end": datetime(2023, 1, 3, 15, 0),
        },
    ]

    result = format_schedule_for_text(raw_schedule)

    assert result.count("\n") == 2

### TESTS build_text
def test_build_text_full_row():
    row = pd.Series({
        "title_fr": "Titre test",
        "description_fr": "<p>Description</p>",
        "longdescription_fr": "<div>Longue description</div>",
        "conditions_fr": "Conditions",
        "keywords_fr": "mot1, mot2",
        "timings": "horaires",
        "location_name": "Lieu",
        "location_description_fr": "Desc lieu",
        "location_address": "Adresse",
        "location_phone": "0102030405",
        "location_website": "site.com",
        "location_links": "lien",
        "location_access_fr": "Accès",
        "location_city": "Paris",
        "location_department": "75",
        "attendancemode": "Offline",
        "onlineaccesslink": "",
        "age_min": 5,
        "age_max": 12
    })

    with patch("build_vectors.clean_html", side_effect=lambda x: x), \
         patch("build_vectors.format_schedule_for_text", return_value="Horaires formatés"):

        result = build_text(row)

    assert "Titre : Titre test" in result
    assert "Description : <p>Description</p>" in result
    assert "Description longue : <div>Longue description</div>" in result
    assert "Horaires :\nHoraires formatés" in result
    assert "Age minimum : 5" in result
    assert "Age maximum : 12" in result

### TESTS embed_batch
class FakeEmbedding:
    def __init__(self, embedding):
        self.embedding = embedding


class FakeResponse:
    def __init__(self, embeddings):
        self.data = [FakeEmbedding(e) for e in embeddings]


def test_embed_batch_success_single_batch():
    texts = ["a", "b", "c"]
    fake_vectors = [[0.1], [0.2], [0.3]]

    fake_client = MagicMock()
    fake_client.embeddings.create.return_value = FakeResponse(fake_vectors)

    with patch("time.sleep"), patch("tqdm.tqdm", lambda x: x):
        result = embed_batch(texts, fake_client, batch_size=10)

    assert result == fake_vectors
    fake_client.embeddings.create.assert_called_once()


def test_embed_batch_multiple_batches():
    texts = ["a", "b", "c", "d"]
    fake_vectors = [[1], [2]]

    fake_client = MagicMock()
    fake_client.embeddings.create.return_value = FakeResponse(fake_vectors)

    with patch("time.sleep"), patch("tqdm.tqdm", lambda x: x):
        result = embed_batch(texts, fake_client, batch_size=2)

    assert result == fake_vectors * 2
    assert fake_client.embeddings.create.call_count == 2


def test_embed_batch_retry_then_success():
    texts = ["a"]
    fake_vectors = [[0.42]]

    fake_client = MagicMock()
    fake_client.embeddings.create.side_effect = [
        Exception("API error"),
        FakeResponse(fake_vectors)
    ]

    with patch("time.sleep"), patch("tqdm.tqdm", lambda x: x):
        result = embed_batch(texts, fake_client, max_retries=2)

    assert result == fake_vectors
    assert fake_client.embeddings.create.call_count == 2


def test_embed_batch_failure_after_max_retries():
    texts = ["a"]
    fake_client = MagicMock()
    fake_client.embeddings.create.side_effect = Exception("API down")

    with patch("time.sleep"), patch("tqdm.tqdm", lambda x: x):
        with pytest.raises(RuntimeError):
            embed_batch(texts, fake_client, max_retries=2)

### TESTS 
def test_build_faiss_indexes_single_column():
    # Données d'entrée
    df = pd.DataFrame({
        "uid": [1, 2],
        "text": ["hello", "world"]
    })

    cols_to_embed = ["text"]
    fake_client = MagicMock()

    # Embeddings factices (2 textes, dim=3)
    fake_vectors = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]

    with patch("build_vectors.embed_batch", return_value=fake_vectors):
        indexes = build_faiss_indexes(df, cols_to_embed, fake_client)

    # Vérifications
    assert isinstance(indexes, dict)
    assert "text" in indexes

    index = indexes["text"]
    assert index.ntotal == 2  # deux vecteurs ajoutés


def test_build_faiss_indexes_multiple_columns():
    df = pd.DataFrame({
        "uid": [10, 20],
        "col1": ["a", "b"],
        "col2": ["c", "d"]
    })

    cols_to_embed = ["col1", "col2"]
    fake_client = MagicMock()

    fake_vectors = [
        [0.0, 0.1],
        [0.2, 0.3]
    ]

    with patch("build_vectors.embed_batch", return_value=fake_vectors):
        indexes = build_faiss_indexes(df, cols_to_embed, fake_client)

    assert set(indexes.keys()) == {"col1", "col2"}

    for index in indexes.values():
        assert index.ntotal == 2

### TESTS save_metadata
def test_save_metadata_creates_correct_pickle(tmp_path):
    # Données d'entrée
    df = pd.DataFrame({
        "id": [1, 2],
        "title": ["Doc 1", "Doc 2"],
        "category": ["A", None]
    })

    cols_meta = ["id", "title", "category"]
    output_path = tmp_path / "metadata.pkl"

    # Appel de la fonction
    result = create_and_save_metadata(df, cols_meta, output_path)

    # Vérification du retour
    expected = [
        {"id": 1, "title": "Doc 1", "category": "A"},
        {"id": 2, "title": "Doc 2", "category": ""}
    ]
    assert result == expected

    # Vérification du fichier pickle
    assert output_path.exists()

    with open(output_path, "rb") as f:
        loaded = pickle.load(f)

    assert loaded == expected

### TESTS save_indexes
def test_save_indexes_creates_directory():
    with patch("os.makedirs") as mock_makedirs, \
         patch("faiss.write_index") as mock_write:

        fake_index = MagicMock()
        indexes = {"test": fake_index}

        save_indexes(indexes, prefix="test_dir")

        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
        mock_write.assert_called_once()


def test_save_indexes_writes_correct_files():
    with patch("faiss.write_index") as mock_write, \
         patch("os.makedirs"):

        fake_index1 = MagicMock()
        fake_index2 = MagicMock()

        indexes = {
            "col1": fake_index1,
            "col2": fake_index2
        }

        save_indexes(indexes, prefix="indexes")

        mock_write.assert_any_call(fake_index1, "indexes/col1.faiss")
        mock_write.assert_any_call(fake_index2, "indexes/col2.faiss")
        assert mock_write.call_count == 2

### TESTS concat_descriptions
def test_concat_descriptions_both_fields():
    row = {
        "description_fr": "<p>Description courte</p>",
        "longdescription_fr": "<div>Description longue</div>"
    }

    result = concat_descriptions(row)

    assert "Description courte" in result
    assert "Description longue" in result


def test_concat_descriptions_only_short():
    row = {
        "description_fr": "<p>Description courte</p>",
        "longdescription_fr": None
    }

    result = concat_descriptions(row)

    assert result == "Description courte"


def test_concat_descriptions_only_long():
    row = {
        "description_fr": None,
        "longdescription_fr": "<div>Description longue</div>"
    }

    result = concat_descriptions(row)

    assert result == "Description longue"


def test_concat_descriptions_no_description():
    row = {
        "description_fr": None,
        "longdescription_fr": None
    }

    result = concat_descriptions(row)

    assert result == ""
    
### TESTS get_date_range
def test_get_date_range_default():
    # Mock de la date du jour pour test déterministe
    fake_today = date(2023, 12, 16)
    with patch("build_vectors.date") as mock_date:
        mock_date.today.return_value = fake_today
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)

        start, end = get_date_range()
        assert start == "2022-12-16"  # 1 an en arrière
        assert end == "2024-12-16"    # 1 an en avant


def test_get_date_range_custom_years():
    fake_today = date(2023, 12, 16)
    with patch("build_vectors.date") as mock_date:
        mock_date.today.return_value = fake_today
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)

        start, end = get_date_range(years_back=2, years_forward=3)
        assert start == "2021-12-16"  # 2 ans en arrière
        assert end == "2026-12-16"    # 3 ans en avant
        
### TESTS fetch_opendatasoft_events
# Mock de la réponse API
def make_fake_response(total_count=1, n_results=1):
    results = [{"uid": i, "title_fr": f"Event {i}"} for i in range(n_results)]
    return {"total_count": total_count, "results": results}


@patch("build_vectors.requests.get")
def test_fetch_opendatasoft_events_single_page(mock_get):
    # Retourne une seule page avec total_count <= 100
    mock_get.return_value.json.return_value = make_fake_response(total_count=50, n_results=50)

    df = fetch_opendatasoft_events(region="TestRegion", start_date="2023-01-01", end_date="2023-12-31")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert "uid" in df.columns
    mock_get.assert_called_once()


@patch("build_vectors.requests.get")
def test_fetch_opendatasoft_events_multiple_pages(mock_get):
    # total_count > 100 → paging
    # Première page
    first_page = make_fake_response(total_count=150, n_results=100)
    # Deuxième page
    second_page = make_fake_response(total_count=150, n_results=50)

    mock_get.side_effect = [
        MagicMock(json=MagicMock(return_value=first_page)),
        MagicMock(json=MagicMock(return_value=second_page))
    ]

    df = fetch_opendatasoft_events(region="TestRegion", start_date="2023-01-01", end_date="2023-12-31")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 150
    assert df.iloc[0]["uid"] == 0
    assert df.iloc[-1]["uid"] == 49  # dernière page uid
    assert mock_get.call_count == 2


@patch("build_vectors.requests.get")
def test_fetch_opendatasoft_events_empty(mock_get):
    # Aucun résultat
    mock_get.return_value.json.return_value = {"total_count": 0, "results": []}

    df = fetch_opendatasoft_events(region="TestRegion", start_date="2023-01-01", end_date="2023-12-31")

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    mock_get.assert_called_once()