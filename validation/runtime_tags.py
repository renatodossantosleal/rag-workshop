"""Apply the workshop tag to supported AWS resource creation calls."""

import functools
import os
import time

import boto3
from botocore.exceptions import ClientError


TAG_KEY = "workshop-kb"
TAG_VALUE = os.environ.get("WORKSHOP_KB_TAG_VALUE", "true")
RAG_TAG_KEY = "rag-workshop"
RAG_TAG_VALUE = os.environ.get("RAG_WORKSHOP_TAG_VALUE", "true")
TAG_MAP = {TAG_KEY: TAG_VALUE, RAG_TAG_KEY: RAG_TAG_VALUE}
TAG_LIST = [
    {"key": TAG_KEY, "value": TAG_VALUE},
    {"key": RAG_TAG_KEY, "value": RAG_TAG_VALUE},
]
TAG_IAM = [
    {"Key": TAG_KEY, "Value": TAG_VALUE},
    {"Key": RAG_TAG_KEY, "Value": RAG_TAG_VALUE},
]


def _merge_tag_map(tags):
    merged = dict(tags or {})
    merged[TAG_KEY] = TAG_VALUE
    merged[RAG_TAG_KEY] = RAG_TAG_VALUE
    return merged


def _merge_tag_list(tags, key_name="key"):
    merged = [
        item
        for item in (tags or [])
        if item.get("key", item.get("Key")) not in {TAG_KEY, RAG_TAG_KEY}
    ]
    value_name = "Value" if key_name == "Key" else "value"
    merged.append({key_name: TAG_KEY, value_name: TAG_VALUE})
    merged.append({key_name: RAG_TAG_KEY, value_name: RAG_TAG_VALUE})
    return merged


def _tag_bucket(client, bucket):
    for attempt in range(5):
        try:
            response = client.get_bucket_tagging(Bucket=bucket)
            tag_set = response.get("TagSet", [])
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code not in {"NoSuchTagSet", "NoSuchBucket", "404", "NotFound"}:
                raise
            tag_set = []

        tags = {
            item["Key"]: item["Value"]
            for item in tag_set
            if item.get("Key") not in {TAG_KEY, RAG_TAG_KEY}
        }
        tags[TAG_KEY] = TAG_VALUE
        tags[RAG_TAG_KEY] = RAG_TAG_VALUE
        try:
            client.put_bucket_tagging(
                Bucket=bucket,
                Tagging={
                    "TagSet": [
                        {"Key": key, "Value": value}
                        for key, value in tags.items()
                    ]
                },
            )
            return
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code not in {"NoSuchBucket", "404", "NotFound"} or attempt == 4:
                raise
            time.sleep(2 ** attempt)


def _find_arn(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower().endswith("arn") and isinstance(item, str):
                return item
            found = _find_arn(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_arn(item)
            if found:
                return found
    return None


class _TaggedClient:
    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        value = getattr(self._client, name)
        if callable(value) and not name.startswith("_"):
            return functools.partial(self._call, name)
        return value

    def _call(self, operation_name, *args, **kwargs):
        service = self._client.meta.service_model.service_name

        if service == "bedrock-agent" and operation_name == "create_knowledge_base":
            kwargs["tags"] = _merge_tag_map(kwargs.get("tags"))
        elif service == "bedrock" and operation_name == "create_guardrail":
            kwargs["tags"] = _merge_tag_list(kwargs.get("tags"))
        elif service == "bedrock-data-automation" and operation_name == "create_data_automation_project":
            kwargs["tags"] = _merge_tag_list(kwargs.get("tags"))
        elif service == "iam" and operation_name in {"create_role", "create_policy"}:
            kwargs["Tags"] = _merge_tag_list(kwargs.get("Tags"), key_name="Key")
        elif service == "lambda" and operation_name == "create_function":
            kwargs["Tags"] = _merge_tag_map(kwargs.get("Tags"))
        elif service == "opensearchserverless" and operation_name == "create_collection":
            kwargs["tags"] = _merge_tag_list(kwargs.get("tags"))
        elif service == "neptune-graph" and operation_name == "create_graph":
            kwargs["tags"] = _merge_tag_map(kwargs.get("tags"))

        response = getattr(self._client, operation_name)(*args, **kwargs)

        if service == "s3" and operation_name == "create_bucket":
            bucket = kwargs.get("Bucket")
            if bucket:
                _tag_bucket(self._client, bucket)
        # The supported create APIs above receive the workshop tag directly.
        # Avoid an immediate follow-up TagResource call: newly created
        # resources, especially BDA projects, can be briefly unavailable to
        # the tagging API while the control plane converges.
        return response

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)


def _wrap_client(client):
    if isinstance(client, _TaggedClient):
        return client
    return _TaggedClient(client)


class _TaggedSession:
    def __init__(self, *args, **kwargs):
        self._session = _ORIGINAL_SESSION(*args, **kwargs)

    def client(self, *args, **kwargs):
        return _wrap_client(self._session.client(*args, **kwargs))

    def resource(self, *args, **kwargs):
        return self._session.resource(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._session, name)


_INSTALLED = False
_ORIGINAL_CLIENT = boto3.client
_ORIGINAL_RESOURCE = boto3.resource
_ORIGINAL_SESSION = boto3.session.Session


def install():
    global _INSTALLED
    if _INSTALLED:
        return

    boto3.client = lambda *args, **kwargs: _wrap_client(
        _ORIGINAL_CLIENT(*args, **kwargs)
    )
    boto3.resource = _ORIGINAL_RESOURCE
    boto3.Session = _TaggedSession
    boto3.session.Session = _TaggedSession
    _INSTALLED = True
