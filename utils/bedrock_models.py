"""Helpers for foundation-model and inference-profile identifiers."""

import boto3


INFERENCE_PROFILE_PREFIXES = ("us.", "global.", "eu.", "apac.")


def model_arn(model_id, region_name=None, account_id=None):
    if model_id.startswith("arn:"):
        return model_id

    session = boto3.session.Session()
    region_name = region_name or session.region_name
    if model_id.startswith(INFERENCE_PROFILE_PREFIXES):
        account_id = account_id or boto3.client("sts").get_caller_identity()["Account"]
        return (
            f"arn:aws:bedrock:{region_name}:{account_id}:"
            f"inference-profile/{model_id}"
        )
    return f"arn:aws:bedrock:{region_name}::foundation-model/{model_id}"


def model_access_arns(model_id, region_name=None, account_id=None):
    """Return the profile ARN and destination model ARNs needed by IAM."""
    primary_arn = model_arn(model_id, region_name, account_id)
    if not model_id.startswith(INFERENCE_PROFILE_PREFIXES):
        return [primary_arn]

    session = boto3.session.Session()
    region_name = region_name or session.region_name
    profile = boto3.client(
        "bedrock",
        region_name=region_name,
    ).get_inference_profile(
        inferenceProfileIdentifier=model_id,
    )
    arns = [primary_arn]
    arns.extend(item["modelArn"] for item in profile.get("models", []))
    return list(dict.fromkeys(arns))
