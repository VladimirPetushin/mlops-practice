# TODO: объявите здесь Airflow DAG hw_basic_batch_inference (extract - infer - publish).
# См. раздел "Задание 1" в ASSIGNMENT.md.
from datetime import timedelta

import pendulum
import requests
from airflow.sdk import DAG, BaseHook, Param, get_current_context, task
from airflow.providers.postgres.hooks.postgres import PostgresHook

import batch_common as core

with DAG(
    dag_id="hw_basic_batch_inference",
    start_date=pendulum.datetime(2026, 9, 1, tz='Europe/Moscow'),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=['homework', 'basic'],

    default_args={
    "retries": 2,
    "retry_delay": timedelta(seconds=5),
    "execution_timeout": timedelta(minutes=2),
    },

    params={
    "business_date": Param("2026-09-01", type="string", format="date"),
    "batch_size": Param(32, type="integer", minimum=1, maximum=1000),
    },

) as dag:
    @task
    def extract():
        context = get_current_context()
        day = core.business_date(
            context["params"]["business_date"],
            context.get("data_interval_start"),
        )
        host = BaseHook.get_connection("source_api").host.rstrip("/")

        response = requests.get(
            f"{host}/basic/{day}.json",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if data["business_date"] != day:
            raise ValueError("Source business_date does not match requested day")

        model_response = requests.get(f"{host}/models/v1.json", timeout=10)
        model_response.raise_for_status()

        return core.save({
            "business_date": day,
            "rows": data["rows"],
            "model": model_response.json(),
            "model_sha256": core.digest(model_response.content),
        })
    @task(retries=0)
    def infer(extracted_ref):
        context = get_current_context()
        extracted = core.read(extracted_ref)

        rows = core.validate_rows(
            extracted["rows"],
            extracted["business_date"],
        )
        predictions = core.infer_batches(
            rows,
            extracted["model"],
            context["params"]["batch_size"],
        )
        core.validate_predictions(
            predictions,
            [row["customer_id"] for row in rows],
        )

        return core.save({
            "business_date": extracted["business_date"],
            "model_version": "v1",
            "predictions": predictions,
            "model_sha256": extracted["model_sha256"],
        })

    @task
    def publish(inferred_ref):
        inferred = core.read(inferred_ref)

        # Передаём свежее соединение: транзакцией управляет replace_day.
        conn = PostgresHook(postgres_conn_id="warehouse").get_conn()
        try:
            core.replace_day(
                conn,
                inferred["business_date"],
                inferred["model_version"],
                inferred["predictions"],
            )
        finally:
            conn.close()

        return {
            "business_date": inferred["business_date"],
            "model_version": inferred["model_version"],
            "rows": len(inferred["predictions"]),
        }

    extracted_ref = extract()
    inferred_ref = infer(extracted_ref)
    publish(inferred_ref)
