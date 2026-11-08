import hashlib
import json
import math
import os
import re
import tempfile
from datetime import date
from pathlib import Path
import logging

import numpy as np

FEATURES = ["age", "tenure_months", "monthly_spend", "support_tickets_30d"]
LIMITS = [(18, 100), (0, 240), (0, 10000), (0, 100)]
ARTIFACTS = Path(os.environ.get("BATCH_ARTIFACTS", "/opt/airflow/include/artifacts"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def save(value):
    content = canonical(value)
    sha = digest(content)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ARTIFACTS, delete=False) as file:
        file.write(content)
        temporary = Path(file.name)
    temporary.replace(ARTIFACTS / f"{sha}.json")
    return {"sha256": sha}


def read(ref):
    sha = ref["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Invalid artifact reference")
    content = (ARTIFACTS / f"{sha}.json").read_bytes()
    if digest(content) != sha:
        raise ValueError("Artifact checksum mismatch")
    return json.loads(content)


def business_date(explicit, interval_start):
    if explicit:
        return date.fromisoformat(explicit).isoformat()
    if interval_start is None:
        raise ValueError("business_date is required without a data interval")
    return interval_start.in_timezone("Europe/Moscow").to_date_string()


def validate_rows(rows, day):
    if len(rows) == 0:
        raise ValueError('rows is empty')
    set_ids = set()
    for row in rows:
        if not isinstance(row['customer_id'], str) or not row['customer_id']:
            raise ValueError("Invalid customer id")
        if row['customer_id'] in set_ids:
            raise ValueError("Duplicate customer_id")
        if row['business_date'] != day:
            raise ValueError("business_date != day")
        if type(row['age']) not in (int, float) or not(row['age'] >= 18 and row['age'] <= 100):
            raise ValueError("Feature is out of range or not int or float")
        elif type(row['tenure_months']) not in (int, float) or not(row['tenure_months'] >= 0 and row['tenure_months'] <= 240):
            raise ValueError("Feature is out of range or not int or float")
        elif type(row['monthly_spend']) not in (int, float) or not(row['monthly_spend'] >= 0 and row['monthly_spend'] <= 10000):
            raise ValueError("Feature is out of range or not int or float")
        elif type(row['support_tickets_30d']) not in (int, float) or not(row['support_tickets_30d'] >= 0 and row['support_tickets_30d'] <= 100):
            raise ValueError("Feature is out of range or not int or float")
        set_ids.add(row['customer_id'])
    return rows


def validate_model(model):
    if model["feature_names"] != FEATURES:
        raise ValueError("Feature order mismatch")
    for field in ("mean", "scale", "weights"):
        if len(model[field]) != len(FEATURES):
            raise ValueError("Model dimensions mismatch")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in model[field]):
            raise ValueError("Invalid model coefficient")
    if any(v <= 0 for v in model["scale"]) or type(model["bias"]) not in (int, float) or not math.isfinite(model["bias"]):
        raise ValueError("Invalid model scale or bias")


def predict(rows, model):
    validate_model(model)
    matrix = np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=np.float64)
    normalized = (matrix - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = normalized @ np.asarray(model["weights"]) + model["bias"]
    scores = np.round(1 / (1 + np.exp(-np.clip(logits, -700, 700))), 8)
    return [{"customer_id": row["customer_id"], "score": float(score), "label": int(score >= 0.5)}
            for row, score in zip(rows, scores)]


def infer_batches(rows, model, batch_size):
    results = []
    if not(type(batch_size) is int):
        raise ValueError('batch_size not int')
    if not(1 <= batch_size <= 1000):
        raise ValueError('batch_size is out of range')
    for i, batch in enumerate(range(0, len(rows), batch_size), start=1):
        logging.info("Batch %d: %d rows", i, len(rows[batch:batch + batch_size]))
        results.extend(predict(rows[batch:batch + batch_size], model))
    return results

def validate_predictions(predictions, expected_ids):
    if len(predictions) != len(expected_ids):
        raise ValueError('len(predictions) != len(expected_ids)')
    seen_ids = set()
    for i, j in zip(predictions, expected_ids):
        if type(i['score']) not in (int, float) or not(0 <= i['score'] <= 1):
            raise ValueError('score is not int or float or out of range')
        if type(i['label']) is not int or i['label'] not in (0, 1):
            raise ValueError('label is not int or not in (0, 1)')
        if not(i['label'] == int(i['score'] >= 0.5)):
            raise ValueError('label does not match score')
        customer_id = i["customer_id"]
        if customer_id in seen_ids:
            raise ValueError("Duplicate customer_id")
        seen_ids.add(i['customer_id'])
    if seen_ids != set(expected_ids):
        raise ValueError('pred_ids do  not match expected_ids')


def replace_day(conn, day, model_version, predictions):
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (day + ":" + model_version,),
            )
            if not predictions:
                raise ValueError("Empty predictions")
            cur.execute(
                """
                DELETE FROM predictions
                WHERE business_date = %s AND model_version = %s
                """,
                (day, model_version)
            )
            for prediction in predictions:
                cur.execute(
                    """
                    INSERT INTO predictions
                        (business_date, model_version, customer_id, score, label)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        day,
                        model_version,
                        prediction["customer_id"],
                        prediction["score"],
                        prediction["label"],
                    ),
                )
    
