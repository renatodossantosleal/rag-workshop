import json
import os
import boto3
import time
from botocore.exceptions import ClientError
import pprint
from retrying import retry
import zipfile
from io import BytesIO
import warnings
import random
warnings.filterwarnings('ignore')

WORKSHOP_TAGS_MAP = {"workshop-kb": "true", "rag-workshop": "true"}
WORKSHOP_TAGS_IAM = [
    {"Key": "workshop-kb", "Value": "true"},
    {"Key": "rag-workshop", "Value": "true"},
]

DEFAULT_GENERATION_MODEL_ID = "anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_EVALUATION_MODEL_ID = "anthropic.claude-sonnet-4-6"
DEFAULT_GENERATION_INFERENCE_PROFILE_ID = (
    "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
DEFAULT_EVALUATION_INFERENCE_PROFILE_ID = "us.anthropic.claude-sonnet-4-6"

valid_generation_models = [
    DEFAULT_GENERATION_MODEL_ID,
    DEFAULT_EVALUATION_MODEL_ID,
    DEFAULT_GENERATION_INFERENCE_PROFILE_ID,
    DEFAULT_EVALUATION_INFERENCE_PROFILE_ID,
    "amazon.nova-micro-v1:0",
]


pp = pprint.PrettyPrinter(indent=2)

def interactive_sleep(seconds: int):
    dots = ''
    for i in range(seconds):
        dots += '.'
        print(dots, end='\r')
        time.sleep(1)

class BedrockManagedKnowledgeBase:
    def __init__(
            self,
            kb_name=None,
            kb_description=None,
            kendra_index=None,
            kbConfigParam = None,
            generation_model=os.getenv("BEDROCK_TEXT_MODEL_ID", DEFAULT_GENERATION_MODEL_ID),
            suffix=None,
    ):
        boto3_session = boto3.session.Session()
        self.region_name = boto3_session.region_name
        self.iam_client = boto3_session.client('iam')
        self.account_number = boto3.client('sts').get_caller_identity().get('Account')
        self.suffix = suffix or f'{self.region_name}-{self.account_number}'
        self.identity = boto3.client('sts').get_caller_identity()['Arn']
        self.bedrock_agent_client = boto3.client('bedrock-agent')
        credentials = boto3.Session().get_credentials()

        self.kb_name = kb_name or f"managed-knowledge-base-{self.suffix}"
        self.kb_description = kb_description or "Managed Knowledge Base"
       
        self.kbConfigParam = kbConfigParam
        self.generation_model = generation_model

        self._validate_models()
        
        self.kb_execution_role_name = f'AmazonBedrockExecutionRoleForManagedKB_{self.suffix}'
        self.fm_policy_name = f'AmazonBedrockFoundationModelPolicyForKnowledgeBase_{self.suffix}'
        self.cw_log_policy_name = f'AmazonBedrockCloudWatchPolicyForKnowledgeBase_{self.suffix}'
        self.kendra_index_policy_name = f'AmazonBedrockKendraIndexPolicyForKnowledgeBase_{self.suffix}'

        self.roles = [self.kb_execution_role_name]

        self.kendra_index =kendra_index

        self._setup_resources()

    def _validate_models(self):
        if self.generation_model not in valid_generation_models:
            raise ValueError(f"Invalid Generation model. Your generation model should be one of {valid_generation_models}")
       

    def _setup_resources(self):

        print("========================================================================================")
        print(f"Step 1 - Creating Knowledge Base Execution Role ({self.kb_execution_role_name}) and Policies")
        
        self.bedrock_kb_execution_role = self.create_bedrock_execution_role_managed_kb()
        self.bedrock_kb_execution_role_name = self.bedrock_kb_execution_role['Role']['RoleName']

        print("========================================================================================")
        print(f"Step 2 - Creating Knowledge Base")
        self.knowledge_base= self.create_managed_knowledge_base()
        print("========================================================================================")

    def create_bedrock_execution_role_managed_kb(self):

        # 0. Create bedrock execution role

        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock.amazonaws.com" 
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        # create bedrock execution role
        try:
            bedrock_kb_execution_role = self.iam_client.create_role(
                RoleName=self.kb_execution_role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
                Description='Amazon Bedrock Knowledge Base Execution Role for accessing Kendra GenAI index',
                MaxSessionDuration=3600,
                Tags=WORKSHOP_TAGS_IAM,
            )
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            bedrock_kb_execution_role = self.iam_client.get_role(
                RoleName=self.kb_execution_role_name
            )

        # fetch arn of the role created above
        bedrock_kb_execution_role_arn = bedrock_kb_execution_role['Role']['Arn']

        
        # 1. Create & attach policy documents for Kendra GenAI Index

        kendra_genai_index_policy_document={
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "VisualEditor0",
                    "Effect": "Allow",
                    "Action": [
                        "kendra:DescribeIndex",
                        "kendra:Retrieve"
                    ],
                    "Resource":f"arn:aws:kendra:{self.region_name}:{self.account_number}:index/*"
                }
            ]
        }
        
        try:
            kendra_genai_index_policy = self.iam_client.create_policy(
                PolicyName=self.kendra_index_policy_name,
                PolicyDocument=json.dumps(kendra_genai_index_policy_document),
                Description='Policy for accessing Kendra GenAI index',
                Tags=WORKSHOP_TAGS_IAM,
            )
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            kendra_genai_index_policy = self.iam_client.get_policy(
                PolicyArn=f"arn:aws:iam::{self.account_number}:policy/"
                f"{self.kendra_index_policy_name}"
            )
    
        # fetch arn of this policy 
        kendra_index_policy_arn = kendra_genai_index_policy["Policy"]["Arn"]
        
        # attach this policy to Amazon Bedrock execution role
        self.iam_client.attach_role_policy(
            RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
            PolicyArn=kendra_index_policy_arn
        )

        return bedrock_kb_execution_role
    
    def _list_knowledge_bases(self):
        summaries = []
        request = {"maxResults": 100}
        while True:
            response = self.bedrock_agent_client.list_knowledge_bases(**request)
            summaries.extend(response.get("knowledgeBaseSummaries", []))
            next_token = response.get("nextToken")
            if not next_token:
                return summaries
            request["nextToken"] = next_token

    def _get_knowledge_base_by_name(self):
        matching = [
            summary for summary in self._list_knowledge_bases()
            if summary.get("name") == self.kb_name
        ]
        if not matching or not matching[0].get("knowledgeBaseId"):
            raise RuntimeError(
                f"Knowledge Base {self.kb_name!r} was not visible after a conflict"
            )
        return self.bedrock_agent_client.get_knowledge_base(
            knowledgeBaseId=matching[0]["knowledgeBaseId"]
        )["knowledgeBase"]

    def _wait_for_knowledge_base_active(self, kb):
        kb_id = kb["knowledgeBaseId"]
        for _ in range(30):
            response = self.bedrock_agent_client.get_knowledge_base(
                knowledgeBaseId=kb_id
            )["knowledgeBase"]
            status = response.get("status")
            if status in {None, "ACTIVE"}:
                return response
            if status in {"FAILED", "DELETE_UNSUCCESSFUL"}:
                raise RuntimeError(
                    f"Knowledge Base {kb_id} is not usable: status={status}"
                )
            time.sleep(2)
        raise TimeoutError(f"Knowledge Base {kb_id} did not become ACTIVE")

    @retry(wait_random_min=1000, wait_random_max=2000, stop_max_attempt_number=7)
    def create_managed_knowledge_base(self):
        try:
            create_kb_response = self.bedrock_agent_client.create_knowledge_base(
                name = self.kb_name,
                description = self.kb_description,
                roleArn = self.bedrock_kb_execution_role['Role']['Arn'],
                knowledgeBaseConfiguration = self.kbConfigParam,
                tags = WORKSHOP_TAGS_MAP,
            )
            kb = create_kb_response["knowledgeBase"]
            pp.pprint(kb)
        except self.bedrock_agent_client.exceptions.ConflictException:
            kb = self._get_knowledge_base_by_name()
            pp.pprint(kb)
        return self._wait_for_knowledge_base_active(kb)
    

    def get_knowledge_base_id(self):
        pp.pprint(self.knowledge_base["knowledgeBaseId"])
        return self.knowledge_base["knowledgeBaseId"]


    def delete_kb(self, delete_iam_roles_and_policies=True):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            # delete KB
            try:
                self.bedrock_agent_client.delete_knowledge_base(
                    knowledgeBaseId=self.knowledge_base['knowledgeBaseId']
                )
                print("======== Knowledge base deleted =========")

            except self.bedrock_agent_client.exceptions.ResourceNotFoundException as e:
                print("Resource not found", e)
            except Exception as e:
                print(e)

            for _ in range(30):
                try:
                    self.bedrock_agent_client.get_knowledge_base(
                        knowledgeBaseId=self.knowledge_base["knowledgeBaseId"]
                    )
                except self.bedrock_agent_client.exceptions.ResourceNotFoundException:
                    break
                time.sleep(2)
            
            # delete role and policies
            if delete_iam_roles_and_policies:
                self.delete_iam_role_and_policies()

    def delete_iam_role_and_policies(self):
        policies_to_detach = []
        marker = None
        while True:
            request = {"RoleName": self.bedrock_kb_execution_role_name}
            if marker:
                request["Marker"] = marker
            response = self.iam_client.list_attached_role_policies(**request)
            policies_to_detach.extend(response.get("AttachedPolicies", []))
            if not response.get("IsTruncated"):
                break
            marker = response.get("Marker")
            if not marker:
                raise RuntimeError(
                    f"IAM returned a truncated policy list without a marker for "
                    f"{self.bedrock_kb_execution_role_name}"
                )

        
        for policy in policies_to_detach:
            policy_name = policy['PolicyName']
            policy_arn = policy['PolicyArn']

            try:
                self.iam_client.detach_role_policy(
                    RoleName=self.kb_execution_role_name,
                    PolicyArn=policy_arn
                )
                if policy_arn.startswith(
                    f"arn:aws:iam::{self.account_number}:policy/"
                ):
                    self.iam_client.delete_policy(PolicyArn=policy_arn)
            except self.iam_client.exceptions.NoSuchEntityException:
                print(f"Policy {policy_arn} not found")

        try:
            self.iam_client.delete_role(RoleName=self.kb_execution_role_name)
        except self.iam_client.exceptions.NoSuchEntityException:
            print(f"Role {self.kb_execution_role_name} not found")
        
        print("======== All IAM roles and policies deleted =========")
