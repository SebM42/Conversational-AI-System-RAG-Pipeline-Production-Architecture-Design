# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 16:45:54 2025

"""

from chatbot_service import embed_text
from chatbot_service import filter_metadata
from chatbot_service import extract_filters_and_query
from chatbot_service import score_and_filter_metadata
from chatbot_service import default_serializer

from unittest.mock import patch, MagicMock
from datetime import datetime, date
import pytest
import numpy as np
import json

### TESTS embed_text
def test_embed_text_success():
    fake_embedding = [0.1, 0.2, 0.3]

    client = MagicMock()
    client.embeddings.create.return_value.data = [
        MagicMock(embedding=fake_embedding)
    ]

    result = embed_text("hello", client)

    assert result == fake_embedding
    client.embeddings.create.assert_called_once()
    
@patch("chatbot_service.time.sleep", return_value=None)
def test_embed_text_retry_then_success(mock_sleep):
    fake_embedding = [0.4, 0.5, 0.6]

    client = MagicMock()

    # 1er appel → exception, 2e appel → succès
    client.embeddings.create.side_effect = [
        Exception("API error"),
        MagicMock(data=[MagicMock(embedding=fake_embedding)])
    ]

    result = embed_text("retry test", client, max_retries=2)

    assert result == fake_embedding
    assert client.embeddings.create.call_count == 2
    
@patch("chatbot_service.time.sleep", return_value=None)
def test_embed_text_max_retries_failure(mock_sleep):
    client = MagicMock()
    client.embeddings.create.side_effect = Exception("API down")

    with pytest.raises(RuntimeError, match="Echec après 3 tentatives"):
        embed_text("fail test", client, max_retries=3)

    assert client.embeddings.create.call_count == 3

### TESTS filter_metadata
@pytest.fixture
def metadata_list():
    return [
        {
            "uid": 1,
            "location_city": "Lyon",
            "location_department": "69",
            "attendancemode": "offline",
            "age_min": 0,
            "age_max": 99
        },
        {
            "uid": 2,
            "location_city": "Saint-Étienne",
            "location_department": "42",
            "attendancemode": "offline",
            "age_min": 12,
            "age_max": 18
        },
        {
            "uid": 3,
            "location_city": "Lyon",
            "location_department": "69",
            "attendancemode": "online",
            "age_min": 18,
            "age_max": 65
        }
    ]

def test_filter_metadata_no_filters(metadata_list):
    filters = {
        "location_city": None,
        "location_department": None,
        "attendancemode": None,
        "age_min": None,
        "age_max": None
    }

    result = filter_metadata(metadata_list, filters)

    assert result == [1, 2, 3]
    
def test_filter_metadata_by_city(metadata_list):
    filters = {
        "location_city": "Lyon",
        "location_department": None,
        "attendancemode": None,
        "age_min": None,
        "age_max": None
    }

    result = filter_metadata(metadata_list, filters)

    assert result == [1, 3]
    
def test_filter_metadata_by_department(metadata_list):
    filters = {
        "location_city": None,
        "location_department": "42",
        "attendancemode": None,
        "age_min": None,
        "age_max": None
    }

    result = filter_metadata(metadata_list, filters)

    assert result == [2]
    
def test_filter_metadata_by_attendancemode(metadata_list):
    filters = {
        "location_city": None,
        "location_department": None,
        "attendancemode": "online",
        "age_min": None,
        "age_max": None
    }

    result = filter_metadata(metadata_list, filters)

    assert result == [3]
    
def test_filter_metadata_by_age_min(metadata_list):
    filters = {
        "location_city": None,
        "location_department": None,
        "attendancemode": None,
        "age_min": 12,
        "age_max": None
    }

    result = filter_metadata(metadata_list, filters)

    assert result == [2]
    
def test_filter_metadata_by_age_max(metadata_list):
    filters = {
        "location_city": None,
        "location_department": None,
        "attendancemode": None,
        "age_min": None,
        "age_max": 65
    }

    result = filter_metadata(metadata_list, filters)

    assert result == [3]
    
def test_filter_metadata_multiple_filters(metadata_list):
    filters = {
        "location_city": "Lyon",
        "location_department": "69",
        "attendancemode": "offline",
        "age_min": 0,
        "age_max": 99
    }

    result = filter_metadata(metadata_list, filters)

    assert result == [1]
    
def test_filter_metadata_no_match(metadata_list):
    filters = {
        "location_city": "Paris",
        "location_department": None,
        "attendancemode": None,
        "age_min": None,
        "age_max": None
    }

    result = filter_metadata(metadata_list, filters)

    assert result == []
    
def test_filter_metadata_missing_fields():
    metadata_list = [
        {"uid": 1},  # champs absents
        {"uid": 2, "location_city": "Lyon"}
    ]

    filters = {
        "location_city": "Lyon",
        "location_department": None,
        "attendancemode": None,
        "age_min": None,
        "age_max": None
    }

    result = filter_metadata(metadata_list, filters)

    assert result == [2]

### TESTS extract_filters_and_query
def test_extract_filters_and_query_with_codeblock():
    client = MagicMock()

    json_response = {
        "filters": {"location_city": "Lyon"},
        "query": "spectacle humour"
    }

    raw_text = f"""
    ```json
    {json.dumps(json_response)}
    ```
    """

    client.chat.complete.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                content=[
                    MagicMock(thinking=[MagicMock(text="thinking")]),
                    MagicMock(text=raw_text)
                ]
            )
        )]
    )

    system_prompt_template = MagicMock()
    system_prompt_template.format.return_value = "system prompt"

    filters, query = extract_filters_and_query(
        client,
        system_prompt_template,
        "Donne-moi des spectacles comiques à Lyon"
    )

    assert filters == {"location_city": "Lyon"}
    assert query == "spectacle humour"
    
def test_extract_filters_and_query_without_codeblock():
    client = MagicMock()

    raw_text = json.dumps({
        "filters": {"location_department": "Loire"},
        "query": "visite château"
    })

    client.chat.complete.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                content=[
                    MagicMock(thinking=[MagicMock(text="thinking")]),
                    MagicMock(text=raw_text)
                ]
            )
        )]
    )

    system_prompt_template = MagicMock()
    system_prompt_template.format.return_value = "system prompt"

    filters, query = extract_filters_and_query(
        client,
        system_prompt_template,
        "Visites de château dans la Loire"
    )

    assert filters == {"location_department": "Loire"}
    assert query == "visite château"

### TESTS score_and_filter_metadata
@patch("chatbot_service.filter_metadata")
def test_score_and_filter_metadata_basic(mock_filter_metadata):
    # Fake docs retournés par la recherche vectorielle
    doc1 = MagicMock()
    doc1.metadata = {"uid": "1", "title": "Event 1"}

    doc2 = MagicMock()
    doc2.metadata = {"uid": "2", "title": "Event 2"}

    vectorstore = MagicMock()
    vectorstore.similarity_search_with_score.return_value = [
        (doc1, 0.1),
        (doc2, 0.8)
    ]

    mock_filter_metadata.return_value = ["1"]

    metadata_list = [{"uid": "1"}, {"uid": "2"}]
    filters = {"location_city": "Lyon"}

    results = score_and_filter_metadata(
        vectorstore=vectorstore,
        metadata_list=metadata_list,
        query_text="concert",
        filters=filters
    )

    assert len(results) == 1
    assert results[0]["uid"] == "1"
    assert results[0]["score"] == 0.1   

@patch("chatbot_service.filter_metadata")
def test_score_and_filter_metadata_no_match(mock_filter_metadata):
    doc = MagicMock()
    doc.metadata = {"uid": "1"}

    vectorstore = MagicMock()
    vectorstore.similarity_search_with_score.return_value = [(doc, 0.5)]

    mock_filter_metadata.return_value = []

    results = score_and_filter_metadata(
        vectorstore,
        metadata_list=[{"uid": "1"}],
        query_text="théâtre",
        filters={}
    )

    assert results == []
    
@patch("chatbot_service.filter_metadata")
def test_similarity_search_called_with_correct_k(mock_filter_metadata):
    vectorstore = MagicMock()
    vectorstore.similarity_search_with_score.return_value = []

    metadata_list = [{"uid": "1"}, {"uid": "2"}, {"uid": "3"}]

    score_and_filter_metadata(
        vectorstore,
        metadata_list,
        query_text="expo",
        filters={}
    )

    vectorstore.similarity_search_with_score.assert_called_once_with(
        "expo",
        k=len(metadata_list)
    )

### TESTS default_serializer
def test_default_serializer_datetime():
    dt = datetime(2024, 1, 15, 14, 30, 0)
    result = default_serializer(dt)

    assert result == "2024-01-15T14:30:00"
    
def test_default_serializer_date():
    d = date(2024, 1, 15)
    result = default_serializer(d)

    assert result == "2024-01-15"
    
def test_default_serializer_numpy_float():
    value = np.float32(0.123)
    result = default_serializer(value)

    assert isinstance(result, float)
    assert result == pytest.approx(0.123)
    
def test_default_serializer_numpy_int():
    value = np.int64(42)
    result = default_serializer(value)

    assert isinstance(result, int)
    assert result == 42
    
def test_default_serializer_numpy_array():
    arr = np.array([1, 2, 3])
    result = default_serializer(arr)

    assert isinstance(result, list)
    assert result == [1, 2, 3]

def test_default_serializer_unsupported_type():
    with pytest.raises(TypeError, match="non sérialisable"):
        default_serializer({"a", "b"})  # set non sérialisable
        
def test_default_serializer_with_json_dumps():
    data = {
        "date": datetime(2024, 1, 1),
        "score": np.float32(0.5),
        "vector": np.array([0.1, 0.2])
    }

    json_str = json.dumps(data, default=default_serializer)

    assert isinstance(json_str, str)